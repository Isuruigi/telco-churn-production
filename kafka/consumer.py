"""
Kafka Consumer for Telco Churn Prediction
Supports both streaming and batch modes
"""
import argparse
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

import config


class TelcoConsumer:
    """
    Kafka consumer for Telco churn prediction.
    Consumes customer data, runs predictions, and publishes results.
    """

    def __init__(
        self,
        bootstrap_servers: list = None,
        input_topic: str = None,
        output_topic: str = None,
        model_path: str = None,
        group_id: str = None
    ):
        """
        Initialize TelcoConsumer

        Args:
            bootstrap_servers: List of Kafka bootstrap servers
            input_topic: Kafka topic to consume from
            output_topic: Kafka topic to produce predictions to
            model_path: Path to trained model
            group_id: Consumer group ID
        """
        self.bootstrap_servers = bootstrap_servers or config.KAFKA_BOOTSTRAP_SERVERS
        self.input_topic = input_topic or config.KAFKA_TOPICS['raw_customers']
        self.output_topic = output_topic or config.KAFKA_TOPICS['predictions']
        self.deadletter_topic = config.KAFKA_TOPICS['deadletter']
        self.model_path = model_path or config.MODEL_CONFIG['model_path']
        self.group_id = group_id or config.CONSUMER_CONFIG['group_id']

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load model
        self.model = self._load_model()

        # Initialize consumer and producer
        self.consumer = None
        self.producer = None

        self.logger.info(
            f"TelcoConsumer initialized - "
            f"Input: {self.input_topic}, Output: {self.output_topic}"
        )

    def _load_model(self):
        """Load trained model from file"""
        try:
            self.logger.info(f"Loading model from: {self.model_path}")
            model = joblib.load(self.model_path)
            self.logger.info("Model loaded successfully")
            return model
        except FileNotFoundError:
            self.logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _create_consumer(self) -> KafkaConsumer:
        """Create and return Kafka consumer instance"""
        try:
            consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id=self.group_id,
                max_poll_records=config.CONSUMER_CONFIG['max_poll_records']
            )
            self.logger.info(f"Kafka consumer created for topic: {self.input_topic}")
            return consumer
        except Exception as e:
            self.logger.error(f"Failed to create Kafka consumer: {e}")
            raise

    def _create_producer(self) -> KafkaProducer:
        """Create and return Kafka producer instance"""
        try:
            producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3
            )
            self.logger.info("Kafka producer created for predictions")
            return producer
        except Exception as e:
            self.logger.error(f"Failed to create Kafka producer: {e}")
            raise

    def _preprocess_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess incoming data for model prediction

        Args:
            data: Raw customer data dictionary

        Returns:
            Preprocessed DataFrame
        """
        try:
            # Remove event_ts and other metadata fields if present
            data_copy = data.copy()
            metadata_fields = ['event_ts', 'processed_ts']
            for field in metadata_fields:
                data_copy.pop(field, None)

            # Create DataFrame
            df = pd.DataFrame([data_copy])

            # Get required features from model
            if hasattr(self.model, 'numerical_features') and hasattr(self.model, 'categorical_features'):
                required_features = self.model.numerical_features + self.model.categorical_features
            else:
                # Fallback to config features
                required_features = config.FEATURE_COLUMNS

            # Ensure all required features are present
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
                # Add missing features with None/NaN
                for feat in missing_features:
                    df[feat] = None

            # Handle data types for numerical features
            if hasattr(self.model, 'numerical_features'):
                numerical_features = self.model.numerical_features
            else:
                numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

            for col in numerical_features:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Handle TotalCharges specifically (common issue in this dataset)
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = df['TotalCharges'].fillna(0)

            # Remove customerID and Churn if present (not used for prediction)
            df = df.drop(columns=[config.DATA_CONFIG['customer_id_field']], errors='ignore')
            df = df.drop(columns=[config.TARGET_COLUMN], errors='ignore')

            return df

        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            raise

    def _predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run prediction on preprocessed data

        Args:
            df: Preprocessed DataFrame

        Returns:
            Prediction results dictionary
        """
        try:
            # Get prediction
            prediction = self.model.predict(df)[0]

            # Get prediction probabilities
            if hasattr(self.model, 'pipeline'):
                # For ChurnPipeline with pipeline attribute
                probabilities = self.model.pipeline.predict_proba(df)[0]
            else:
                # Direct sklearn pipeline
                probabilities = self.model.predict_proba(df)[0]

            # Assuming binary classification: [prob_no_churn, prob_churn]
            churn_probability = float(probabilities[1] if len(probabilities) > 1 else probabilities[0])

            result = {
                'prediction': prediction,
                'churn_probability': churn_probability,
                'confidence_score': float(np.max(probabilities))
            }

            return result

        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

    def _create_prediction_message(
        self,
        customer_id: str,
        prediction_result: Dict[str, Any],
        original_event_ts: str
    ) -> Dict[str, Any]:
        """
        Create prediction message for output topic

        Args:
            customer_id: Customer ID
            prediction_result: Prediction results from model
            original_event_ts: Original event timestamp

        Returns:
            Formatted prediction message
        """
        # Convert numpy types to Python native types for JSON serialization
        prediction = prediction_result['prediction']
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        elif isinstance(prediction, (np.integer, np.floating)):
            prediction = prediction.item()

        return {
            'customerID': str(customer_id),
            'churn_probability': float(prediction_result['churn_probability']),
            'prediction': str(prediction),
            'confidence_score': float(prediction_result.get('confidence_score', 0)),
            'event_ts': str(original_event_ts),
            'processed_ts': datetime.now().isoformat()
        }

    def _send_prediction(self, prediction_message: Dict[str, Any], key: str) -> bool:
        """
        Send prediction to output topic

        Args:
            prediction_message: Prediction message
            key: Message key (customerID)

        Returns:
            True if successful, False otherwise
        """
        try:
            future = self.producer.send(
                self.output_topic,
                key=key,
                value=prediction_message
            )
            future.get(timeout=10)
            self.logger.debug(f"Prediction sent for customer: {key}")
            return True

        except KafkaError as e:
            self.logger.error(f"Failed to send prediction for {key}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending prediction: {e}")
            return False

    def _send_to_deadletter(
        self,
        message: Dict[str, Any],
        key: str,
        error: str,
        stage: str = "processing"
    ):
        """Send failed message to dead letter queue"""
        try:
            deadletter_message = {
                'original_message': message,
                'error': error,
                'stage': stage,
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

    def stream_mode(self, duration: Optional[int] = None):
        """
        Streaming mode: continuously consume and process messages

        Args:
            duration: Optional duration in seconds (None for infinite)
        """
        self.logger.info("Starting streaming mode")

        # Create consumer and producer
        self.consumer = self._create_consumer()
        self.producer = self._create_producer()

        import time
        start_time = time.time()
        messages_processed = 0
        predictions_sent = 0
        errors = 0

        try:
            for message in self.consumer:
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    self.logger.info(f"Duration limit reached: {duration}s")
                    break

                try:
                    # Extract message data
                    customer_data = message.value
                    customer_id = message.key or customer_data.get(config.DATA_CONFIG['customer_id_field'])
                    original_event_ts = customer_data.get('event_ts', datetime.now().isoformat())

                    messages_processed += 1

                    # Preprocess data
                    df = self._preprocess_data(customer_data)

                    # Run prediction
                    prediction_result = self._predict(df)

                    # Create prediction message
                    prediction_message = self._create_prediction_message(
                        customer_id,
                        prediction_result,
                        original_event_ts
                    )

                    # Send prediction
                    if self._send_prediction(prediction_message, customer_id):
                        predictions_sent += 1
                    else:
                        errors += 1

                    # Log progress every 100 messages
                    if messages_processed % 100 == 0:
                        self.logger.info(
                            f"Progress: {messages_processed} processed, "
                            f"{predictions_sent} predictions sent, {errors} errors"
                        )

                except Exception as e:
                    errors += 1
                    self.logger.error(f"Error processing message: {e}")
                    # Send to dead letter queue
                    if message.key:
                        self._send_to_deadletter(
                            message.value,
                            message.key,
                            str(e),
                            "prediction"
                        )

        except KeyboardInterrupt:
            self.logger.info("Streaming interrupted by user")
        finally:
            self.logger.info(
                f"Streaming completed: {messages_processed} processed, "
                f"{predictions_sent} predictions sent, {errors} errors"
            )
            self.close()

    def batch_mode(self, window_size: int = 100, num_windows: Optional[int] = None):
        """
        Batch mode: consume defined window, run batch predictions

        Args:
            window_size: Number of messages per batch
            num_windows: Number of windows to process (None for all available)
        """
        self.logger.info(
            f"Starting batch mode: window_size={window_size}, "
            f"num_windows={num_windows or 'all'}"
        )

        # Create consumer and producer
        self.consumer = self._create_consumer()
        self.producer = self._create_producer()

        windows_processed = 0
        total_messages = 0
        total_predictions = 0
        total_errors = 0

        try:
            while True:
                # Check window limit
                if num_windows and windows_processed >= num_windows:
                    self.logger.info(f"Window limit reached: {num_windows}")
                    break

                # Collect batch
                batch_messages = []
                batch_data = []
                batch_keys = []

                self.logger.info(f"Collecting window {windows_processed + 1}...")

                # Poll messages for this window
                message_count = 0
                poll_timeout_ms = 5000
                empty_polls = 0
                max_empty_polls = 3

                while message_count < window_size:
                    messages = self.consumer.poll(timeout_ms=poll_timeout_ms, max_records=window_size)

                    if not messages:
                        empty_polls += 1
                        if empty_polls >= max_empty_polls:
                            self.logger.info("No more messages available")
                            break
                        continue

                    empty_polls = 0

                    for topic_partition, records in messages.items():
                        for message in records:
                            batch_messages.append(message)
                            batch_data.append(message.value)
                            batch_keys.append(
                                message.key or message.value.get(config.DATA_CONFIG['customer_id_field'])
                            )
                            message_count += 1

                            if message_count >= window_size:
                                break
                        if message_count >= window_size:
                            break

                # If no messages collected, exit
                if not batch_messages:
                    self.logger.info("No messages in this window, exiting batch mode")
                    break

                self.logger.info(f"Processing window with {len(batch_messages)} messages")

                # Process batch
                window_predictions = 0
                window_errors = 0

                for i, (message_data, key) in enumerate(zip(batch_data, batch_keys)):
                    try:
                        # Preprocess
                        df = self._preprocess_data(message_data)

                        # Predict
                        prediction_result = self._predict(df)

                        # Create prediction message
                        original_event_ts = message_data.get('event_ts', datetime.now().isoformat())
                        prediction_message = self._create_prediction_message(
                            key,
                            prediction_result,
                            original_event_ts
                        )

                        # Send prediction
                        if self._send_prediction(prediction_message, key):
                            window_predictions += 1
                        else:
                            window_errors += 1

                    except Exception as e:
                        window_errors += 1
                        self.logger.error(f"Error processing message in batch: {e}")
                        self._send_to_deadletter(message_data, key, str(e), "batch_prediction")

                # Flush producer after window
                self.producer.flush()

                # Update counters
                windows_processed += 1
                total_messages += len(batch_messages)
                total_predictions += window_predictions
                total_errors += window_errors

                # Print window summary
                self.logger.info(
                    f"Window {windows_processed} complete: "
                    f"{len(batch_messages)} messages, "
                    f"{window_predictions} predictions, "
                    f"{window_errors} errors"
                )

        except KeyboardInterrupt:
            self.logger.info("Batch processing interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
        finally:
            # Print final summary
            self.logger.info("=" * 60)
            self.logger.info("Batch Processing Summary:")
            self.logger.info(f"  Total windows processed: {windows_processed}")
            self.logger.info(f"  Total messages: {total_messages}")
            self.logger.info(f"  Total predictions: {total_predictions}")
            self.logger.info(f"  Total errors: {total_errors}")
            if total_messages > 0:
                success_rate = (total_predictions / total_messages) * 100
                self.logger.info(f"  Success rate: {success_rate:.2f}%")
            self.logger.info("=" * 60)
            self.close()

    def close(self):
        """Close consumer and producer"""
        if self.consumer:
            self.logger.info("Closing consumer...")
            self.consumer.close()
        if self.producer:
            self.logger.info("Flushing and closing producer...")
            self.producer.flush()
            self.producer.close()
        self.logger.info("Consumer closed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Telco Churn Kafka Consumer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['streaming', 'batch'],
        required=True,
        help='Consumer mode: streaming or batch'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Duration in seconds for streaming mode (default: infinite)'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=100,
        help='Window size for batch mode (default: 100)'
    )

    parser.add_argument(
        '--num-windows',
        type=int,
        default=None,
        help='Number of windows to process in batch mode (default: all available)'
    )

    parser.add_argument(
        '--group-id',
        type=str,
        default=None,
        help=f"Consumer group ID (default: {config.CONSUMER_CONFIG['group_id']})"
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

    # Validate model exists
    if not config.MODEL_CONFIG['model_path'].exists():
        logging.error(f"Model not found at: {config.MODEL_CONFIG['model_path']}")
        logging.error("Please ensure the model file exists before running the consumer")
        return

    # Create consumer
    consumer = TelcoConsumer(group_id=args.group_id)

    # Run in selected mode
    if args.mode == 'streaming':
        consumer.stream_mode(duration=args.duration)
    elif args.mode == 'batch':
        consumer.batch_mode(
            window_size=args.window_size,
            num_windows=args.num_windows
        )


if __name__ == "__main__":
    main()
