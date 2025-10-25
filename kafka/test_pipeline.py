"""
Test script for the full Kafka pipeline
Runs producer and consumer in separate threads
"""
import argparse
import logging
import threading
import time
from datetime import datetime
from kafka import KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError

import config
from producer import TelcoProducer
from consumer import TelcoConsumer


class PipelineTester:
    """Test the full Kafka pipeline"""

    def __init__(self, log_level: str = 'INFO'):
        """Initialize the pipeline tester"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, log_level))

        self.producer_thread = None
        self.consumer_thread = None
        self.producer_done = threading.Event()
        self.consumer_done = threading.Event()

    def verify_kafka_connection(self):
        """Verify Kafka is accessible"""
        try:
            self.logger.info("Verifying Kafka connection...")
            admin_client = KafkaAdminClient(
                bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS
            )
            admin_client.close()
            self.logger.info("✓ Kafka connection verified")
            return True
        except Exception as e:
            self.logger.error(f"✗ Failed to connect to Kafka: {e}")
            return False

    def verify_topics(self, create_if_missing: bool = True):
        """
        Verify required topics exist

        Args:
            create_if_missing: Create topics if they don't exist
        """
        try:
            self.logger.info("Verifying Kafka topics...")
            admin_client = KafkaAdminClient(
                bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS
            )

            existing_topics = admin_client.list_topics()
            required_topics = list(config.KAFKA_TOPICS.values())

            missing_topics = [t for t in required_topics if t not in existing_topics]

            if missing_topics:
                if create_if_missing:
                    self.logger.info(f"Creating missing topics: {missing_topics}")
                    new_topics = [
                        NewTopic(name=topic, num_partitions=1, replication_factor=1)
                        for topic in missing_topics
                    ]
                    try:
                        admin_client.create_topics(new_topics=new_topics)
                        self.logger.info("✓ Topics created successfully")
                    except TopicAlreadyExistsError:
                        self.logger.info("✓ Topics already exist")
                else:
                    self.logger.warning(f"Missing topics: {missing_topics}")
                    admin_client.close()
                    return False

            self.logger.info("✓ All required topics exist")
            admin_client.close()
            return True

        except Exception as e:
            self.logger.error(f"✗ Failed to verify topics: {e}")
            return False

    def verify_model(self):
        """Verify model file exists"""
        model_path = config.MODEL_CONFIG['model_path']
        if model_path.exists():
            self.logger.info(f"✓ Model file found: {model_path}")
            return True
        else:
            self.logger.error(f"✗ Model file not found: {model_path}")
            return False

    def verify_data(self):
        """Verify CSV data file exists"""
        csv_path = config.DATA_CONFIG['csv_path']
        if csv_path.exists():
            self.logger.info(f"✓ CSV file found: {csv_path}")
            return True
        else:
            self.logger.error(f"✗ CSV file not found: {csv_path}")
            return False

    def run_producer(
        self,
        mode: str = 'streaming',
        events_per_sec: int = 10,
        duration: int = 30,
        batch_size: int = 100
    ):
        """
        Run producer in a separate thread

        Args:
            mode: 'streaming' or 'batch'
            events_per_sec: Events per second for streaming mode
            duration: Duration for streaming mode
            batch_size: Batch size for batch mode
        """
        try:
            self.logger.info(f"Starting producer in {mode} mode")
            producer = TelcoProducer()

            if mode == 'streaming':
                producer.stream_mode(
                    events_per_sec=events_per_sec,
                    duration=duration
                )
            elif mode == 'batch':
                producer.batch_mode(batch_size=batch_size)

            self.logger.info("Producer completed")
        except Exception as e:
            self.logger.error(f"Producer error: {e}")
        finally:
            self.producer_done.set()

    def run_consumer(
        self,
        mode: str = 'streaming',
        duration: int = 35,
        window_size: int = 100,
        num_windows: int = None
    ):
        """
        Run consumer in a separate thread

        Args:
            mode: 'streaming' or 'batch'
            duration: Duration for streaming mode
            window_size: Window size for batch mode
            num_windows: Number of windows for batch mode
        """
        try:
            self.logger.info(f"Starting consumer in {mode} mode")
            consumer = TelcoConsumer()

            if mode == 'streaming':
                consumer.stream_mode(duration=duration)
            elif mode == 'batch':
                consumer.batch_mode(
                    window_size=window_size,
                    num_windows=num_windows
                )

            self.logger.info("Consumer completed")
        except Exception as e:
            self.logger.error(f"Consumer error: {e}")
        finally:
            self.consumer_done.set()

    def verify_predictions(self, timeout: int = 60):
        """
        Verify predictions are being published

        Args:
            timeout: Timeout in seconds
        """
        try:
            self.logger.info("Verifying predictions are published...")

            consumer = KafkaConsumer(
                config.KAFKA_TOPICS['predictions'],
                bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
                auto_offset_reset='latest',
                consumer_timeout_ms=timeout * 1000,
                value_deserializer=lambda v: v.decode('utf-8')
            )

            prediction_count = 0
            start_time = time.time()

            for message in consumer:
                prediction_count += 1
                if prediction_count == 1:
                    self.logger.info(f"✓ First prediction received: {message.value[:100]}...")

                if prediction_count >= 5:
                    break

                if time.time() - start_time > timeout:
                    break

            consumer.close()

            if prediction_count > 0:
                self.logger.info(f"✓ Verified {prediction_count} predictions")
                return True
            else:
                self.logger.warning("✗ No predictions received")
                return False

        except Exception as e:
            self.logger.error(f"✗ Failed to verify predictions: {e}")
            return False

    def test_streaming_mode(
        self,
        events_per_sec: int = 10,
        duration: int = 30
    ):
        """
        Test streaming mode

        Args:
            events_per_sec: Events per second
            duration: Duration in seconds
        """
        self.logger.info("=" * 60)
        self.logger.info("TESTING STREAMING MODE")
        self.logger.info("=" * 60)

        # Start consumer first
        self.consumer_thread = threading.Thread(
            target=self.run_consumer,
            kwargs={
                'mode': 'streaming',
                'duration': duration + 10  # Run a bit longer than producer
            }
        )
        self.consumer_thread.start()

        # Give consumer time to start
        time.sleep(3)

        # Start producer
        self.producer_thread = threading.Thread(
            target=self.run_producer,
            kwargs={
                'mode': 'streaming',
                'events_per_sec': events_per_sec,
                'duration': duration
            }
        )
        self.producer_thread.start()

        # Wait for producer to finish
        self.logger.info("Waiting for producer to complete...")
        self.producer_done.wait()

        # Give consumer a bit more time to process remaining messages
        time.sleep(5)

        # Wait for consumer to finish
        self.logger.info("Waiting for consumer to complete...")
        self.consumer_done.wait(timeout=30)

        self.logger.info("=" * 60)
        self.logger.info("STREAMING MODE TEST COMPLETED")
        self.logger.info("=" * 60)

    def test_batch_mode(
        self,
        batch_size: int = 50,
        num_batches: int = 2
    ):
        """
        Test batch mode

        Args:
            batch_size: Batch size
            num_batches: Number of batches
        """
        self.logger.info("=" * 60)
        self.logger.info("TESTING BATCH MODE")
        self.logger.info("=" * 60)

        # Start consumer first
        self.consumer_thread = threading.Thread(
            target=self.run_consumer,
            kwargs={
                'mode': 'batch',
                'window_size': batch_size,
                'num_windows': num_batches
            }
        )
        self.consumer_thread.start()

        # Give consumer time to start
        time.sleep(3)

        # Start producer
        total_messages = batch_size * num_batches
        self.producer_thread = threading.Thread(
            target=self.run_producer,
            kwargs={
                'mode': 'batch',
                'batch_size': total_messages
            }
        )
        self.producer_thread.start()

        # Wait for both to finish
        self.logger.info("Waiting for producer to complete...")
        self.producer_done.wait()

        self.logger.info("Waiting for consumer to complete...")
        self.consumer_done.wait(timeout=60)

        self.logger.info("=" * 60)
        self.logger.info("BATCH MODE TEST COMPLETED")
        self.logger.info("=" * 60)

    def run_full_test(self, mode: str = 'streaming'):
        """
        Run full pipeline test

        Args:
            mode: 'streaming' or 'batch'
        """
        self.logger.info("=" * 60)
        self.logger.info("KAFKA PIPELINE FULL TEST")
        self.logger.info(f"Mode: {mode}")
        self.logger.info(f"Timestamp: {datetime.now()}")
        self.logger.info("=" * 60)

        # Verify prerequisites
        self.logger.info("\nStep 1: Verifying prerequisites...")
        if not self.verify_kafka_connection():
            self.logger.error("Kafka connection failed. Exiting.")
            return False

        if not self.verify_topics(create_if_missing=True):
            self.logger.error("Topic verification failed. Exiting.")
            return False

        if not self.verify_model():
            self.logger.error("Model verification failed. Exiting.")
            return False

        if not self.verify_data():
            self.logger.error("Data verification failed. Exiting.")
            return False

        self.logger.info("\n✓ All prerequisites verified\n")

        # Run test
        self.logger.info("Step 2: Running pipeline test...")
        if mode == 'streaming':
            self.test_streaming_mode(events_per_sec=10, duration=20)
        elif mode == 'batch':
            self.test_batch_mode(batch_size=50, num_batches=2)

        # Verify predictions
        self.logger.info("\nStep 3: Verifying predictions...")
        self.verify_predictions(timeout=30)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("PIPELINE TEST COMPLETED")
        self.logger.info("=" * 60)

        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Test Telco Churn Kafka Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['streaming', 'batch', 'both'],
        default='streaming',
        help='Test mode: streaming, batch, or both (default: streaming)'
    )

    parser.add_argument(
        '--events-per-sec',
        type=int,
        default=10,
        help='Events per second for streaming mode (default: 10)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=20,
        help='Duration in seconds for streaming mode (default: 20)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for batch mode (default: 50)'
    )

    parser.add_argument(
        '--num-batches',
        type=int,
        default=2,
        help='Number of batches for batch mode (default: 2)'
    )

    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify prerequisites without running the pipeline'
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

    # Create tester
    tester = PipelineTester(log_level=args.log_level)

    # Verify prerequisites
    if args.verify_only:
        print("\nVerifying prerequisites only...\n")
        tester.verify_kafka_connection()
        tester.verify_topics(create_if_missing=False)
        tester.verify_model()
        tester.verify_data()
        return

    # Run tests
    if args.mode == 'both':
        # Test streaming mode
        tester.run_full_test(mode='streaming')

        # Reset events
        tester.producer_done.clear()
        tester.consumer_done.clear()

        # Test batch mode
        print("\n\n")
        tester.run_full_test(mode='batch')

    else:
        tester.run_full_test(mode=args.mode)


if __name__ == "__main__":
    main()
