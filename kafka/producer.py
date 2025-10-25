"""
Kafka Producer for Telco Churn Prediction
Supports both streaming and batch modes
"""
import argparse
import json
import logging
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from kafka import KafkaProducer
from kafka.errors import KafkaError

import config


class TelcoProducer:
    """
    Kafka producer for Telco customer data.
    Supports streaming (continuous) and batch modes.
    """

    def __init__(
        self,
        bootstrap_servers: list = None,
        topic: str = None
    ):
        """
        Initialize TelcoProducer

        Args:
            bootstrap_servers: List of Kafka bootstrap servers
            topic: Kafka topic to produce to
        """
        self.bootstrap_servers = bootstrap_servers or config.KAFKA_BOOTSTRAP_SERVERS
        self.topic = topic or config.KAFKA_TOPICS['raw_customers']
        self.deadletter_topic = config.KAFKA_TOPICS['deadletter']

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize producer
        self.producer = self._create_producer()

        # Load CSV data
        self.csv_path = config.DATA_CONFIG['csv_path']
        self.customer_id_field = config.DATA_CONFIG['customer_id_field']
        self.df = None

        self.logger.info(f"TelcoProducer initialized with topic: {self.topic}")

    def _create_producer(self) -> KafkaProducer:
        """Create and return Kafka producer instance"""
        try:
            producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            self.logger.info("Kafka producer created successfully")
            return producer
        except Exception as e:
            self.logger.error(f"Failed to create Kafka producer: {e}")
            raise

    def load_data(self):
        """Load CSV data into DataFrame"""
        try:
            self.logger.info(f"Loading data from: {self.csv_path}")
            self.df = pd.read_csv(self.csv_path)
            self.logger.info(f"Loaded {len(self.df)} records from CSV")
            return self.df
        except Exception as e:
            self.logger.error(f"Failed to load CSV data: {e}")
            raise

    def _prepare_message(self, row: pd.Series) -> Dict[str, Any]:
        """
        Prepare message from DataFrame row

        Args:
            row: pandas Series representing a row

        Returns:
            Dictionary with message data
        """
        # Convert row to dict and handle NaN values
        message = row.to_dict()

        # Replace NaN with None for JSON serialization
        for key, value in message.items():
            if pd.isna(value):
                message[key] = None
            # Convert numpy types to Python types
            elif hasattr(value, 'item'):
                message[key] = value.item()

        # Add event timestamp
        message['event_ts'] = datetime.now().isoformat()

        return message

    def _send_message(self, message: Dict[str, Any], key: str) -> bool:
        """
        Send message to Kafka topic

        Args:
            message: Message dictionary
            key: Message key (customerID)

        Returns:
            True if successful, False otherwise
        """
        try:
            future = self.producer.send(
                self.topic,
                key=key,
                value=message
            )
            # Wait for message to be sent (with timeout)
            record_metadata = future.get(timeout=10)

            self.logger.debug(
                f"Message sent to {record_metadata.topic} "
                f"partition {record_metadata.partition} "
                f"offset {record_metadata.offset}"
            )
            return True

        except KafkaError as e:
            self.logger.error(f"Failed to send message for key {key}: {e}")
            # Send to dead letter queue
            self._send_to_deadletter(message, key, str(e))
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending message: {e}")
            self._send_to_deadletter(message, key, str(e))
            return False

    def _send_to_deadletter(self, message: Dict[str, Any], key: str, error: str):
        """Send failed message to dead letter queue"""
        try:
            deadletter_message = {
                'original_message': message,
                'error': error,
                'failed_at': datetime.now().isoformat()
            }
            self.producer.send(
                self.deadletter_topic,
                key=key,
                value=deadletter_message
            )
            self.logger.info(f"Message sent to dead letter queue: {key}")
        except Exception as e:
            self.logger.error(f"Failed to send to dead letter queue: {e}")

    def stream_mode(self, events_per_sec: int = 10, duration: Optional[int] = None):
        """
        Stream mode: continuously sample rows and send to Kafka

        Args:
            events_per_sec: Number of events to send per second
            duration: Optional duration in seconds (None for infinite)
        """
        if self.df is None:
            self.load_data()

        self.logger.info(
            f"Starting streaming mode: {events_per_sec} events/sec"
            + (f" for {duration}s" if duration else " (continuous)")
        )

        sleep_interval = 1.0 / events_per_sec
        start_time = time.time()
        messages_sent = 0
        messages_failed = 0

        try:
            while True:
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    self.logger.info(f"Duration limit reached: {duration}s")
                    break

                # Sample random row
                row = self.df.sample(n=1).iloc[0]
                message = self._prepare_message(row)
                key = str(row[self.customer_id_field])

                # Send message
                if self._send_message(message, key):
                    messages_sent += 1
                else:
                    messages_failed += 1

                # Log progress every 100 messages
                if (messages_sent + messages_failed) % 100 == 0:
                    self.logger.info(
                        f"Progress: {messages_sent} sent, {messages_failed} failed"
                    )

                # Sleep to control rate
                time.sleep(sleep_interval)

        except KeyboardInterrupt:
            self.logger.info("Streaming interrupted by user")
        finally:
            self.logger.info(
                f"Streaming completed: {messages_sent} sent, {messages_failed} failed"
            )
            self.close()

    def batch_mode(
        self,
        batch_size: int = 100,
        resume: bool = False,
        checkpoint_file: str = "producer_checkpoint.txt"
    ):
        """
        Batch mode: send CSV in chunks with checkpoint/resume support

        Args:
            batch_size: Number of records per batch
            resume: Whether to resume from checkpoint
            checkpoint_file: Path to checkpoint file
        """
        if self.df is None:
            self.load_data()

        checkpoint_path = Path(config.BATCH_CONFIG['checkpoint_dir']) / checkpoint_file
        start_index = 0

        # Resume from checkpoint if requested
        if resume and checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    start_index = int(f.read().strip())
                self.logger.info(f"Resuming from checkpoint: index {start_index}")
            except Exception as e:
                self.logger.warning(f"Failed to read checkpoint: {e}, starting from beginning")
                start_index = 0

        self.logger.info(
            f"Starting batch mode: batch_size={batch_size}, "
            f"starting from index {start_index}"
        )

        total_records = len(self.df)
        messages_sent = 0
        messages_failed = 0
        current_index = start_index

        try:
            # Process in batches
            while current_index < total_records:
                batch_end = min(current_index + batch_size, total_records)
                batch = self.df.iloc[current_index:batch_end]

                self.logger.info(
                    f"Processing batch: {current_index} to {batch_end} "
                    f"({len(batch)} records)"
                )

                # Send each record in batch
                for idx, row in batch.iterrows():
                    message = self._prepare_message(row)
                    key = str(row[self.customer_id_field])

                    if self._send_message(message, key):
                        messages_sent += 1
                    else:
                        messages_failed += 1

                # Flush after each batch
                self.producer.flush()

                # Update checkpoint
                current_index = batch_end
                try:
                    checkpoint_path.parent.mkdir(exist_ok=True)
                    with open(checkpoint_path, 'w') as f:
                        f.write(str(current_index))
                    self.logger.debug(f"Checkpoint updated: {current_index}")
                except Exception as e:
                    self.logger.warning(f"Failed to update checkpoint: {e}")

                # Log progress
                progress_pct = (current_index / total_records) * 100
                self.logger.info(
                    f"Progress: {current_index}/{total_records} "
                    f"({progress_pct:.1f}%) - "
                    f"{messages_sent} sent, {messages_failed} failed"
                )

        except KeyboardInterrupt:
            self.logger.info("Batch processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
        finally:
            self.logger.info(
                f"Batch processing completed: {messages_sent} sent, "
                f"{messages_failed} failed, stopped at index {current_index}"
            )
            self.close()

    def close(self):
        """Close producer and cleanup"""
        if self.producer:
            self.logger.info("Flushing remaining messages...")
            self.producer.flush()
            self.producer.close()
            self.logger.info("Producer closed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Telco Churn Kafka Producer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['streaming', 'batch'],
        required=True,
        help='Producer mode: streaming or batch'
    )

    parser.add_argument(
        '--events-per-sec',
        type=int,
        default=config.STREAMING_CONFIG['default_events_per_sec'],
        help=f"Events per second for streaming mode (default: {config.STREAMING_CONFIG['default_events_per_sec']})"
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Duration in seconds for streaming mode (default: infinite)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=config.BATCH_CONFIG['default_batch_size'],
        help=f"Batch size for batch mode (default: {config.BATCH_CONFIG['default_batch_size']})"
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint in batch mode'
    )

    parser.add_argument(
        '--checkpoint-file',
        type=str,
        default='producer_checkpoint.txt',
        help='Checkpoint file name (default: producer_checkpoint.txt)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create directories
    config.create_directories()

    # Create producer
    producer = TelcoProducer()

    # Run in selected mode
    if args.mode == 'streaming':
        producer.stream_mode(
            events_per_sec=args.events_per_sec,
            duration=args.duration
        )
    elif args.mode == 'batch':
        producer.batch_mode(
            batch_size=args.batch_size,
            resume=args.resume,
            checkpoint_file=args.checkpoint_file
        )


if __name__ == "__main__":
    main()
