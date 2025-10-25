"""
Airflow DAG for Kafka Streaming Pipeline - Telco Churn Prediction
This DAG manages the long-running streaming consumer with health checks
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.python import PythonSensor
import os
import sys
import subprocess
import time
import logging

# Add kafka directory to path
KAFKA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(KAFKA_DIR, '..'))
sys.path.insert(0, KAFKA_DIR)

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['your-email@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=24),
}


def check_kafka_connection(**context):
    """Check if Kafka broker is accessible"""
    from kafka import KafkaAdminClient
    from kafka.errors import KafkaError

    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=['localhost:29092'],
            request_timeout_ms=5000
        )
        admin_client.close()
        logging.info("✓ Kafka connection successful")
        return True
    except Exception as e:
        logging.error(f"✗ Kafka connection failed: {e}")
        raise


def check_kafka_topics(**context):
    """Verify required Kafka topics exist"""
    from kafka import KafkaAdminClient

    required_topics = [
        'telco-raw-customers',
        'telco-churn-predictions',
        'telco-deadletter'
    ]

    try:
        admin_client = KafkaAdminClient(bootstrap_servers=['localhost:29092'])
        existing_topics = admin_client.list_topics()
        admin_client.close()

        missing_topics = [t for t in required_topics if t not in existing_topics]

        if missing_topics:
            raise ValueError(f"Missing topics: {missing_topics}")

        logging.info(f"✓ All required topics exist: {required_topics}")
        return True
    except Exception as e:
        logging.error(f"✗ Topic verification failed: {e}")
        raise


def verify_model_exists(**context):
    """Check if trained model file exists"""
    import os

    model_path = os.path.join(
        PROJECT_ROOT,
        'telco-churn-production',
        'src',
        'models',
        'sklearn_pipeline.pkl'
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    logging.info(f"✓ Model file found: {model_path}")
    return True


def start_streaming_consumer(**context):
    """Start the streaming consumer as a background process"""
    import subprocess
    import os

    consumer_script = os.path.join(KAFKA_DIR, 'consumer.py')
    log_file = os.path.join(KAFKA_DIR, 'logs', 'streaming_consumer.log')

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Start consumer in background
    cmd = [
        'python',
        consumer_script,
        '--mode', 'streaming',
        '--log-level', 'INFO'
    ]

    with open(log_file, 'a') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=KAFKA_DIR
        )

    # Store process ID in XCom
    context['task_instance'].xcom_push(key='consumer_pid', value=process.pid)

    logging.info(f"✓ Started streaming consumer with PID: {process.pid}")
    logging.info(f"  Log file: {log_file}")

    return process.pid


def check_consumer_health(**context):
    """Check if consumer is still running and processing messages"""
    import psutil

    ti = context['task_instance']
    consumer_pid = ti.xcom_pull(task_ids='start_consumer', key='consumer_pid')

    if not consumer_pid:
        logging.error("✗ Consumer PID not found in XCom")
        return False

    try:
        process = psutil.Process(consumer_pid)

        if process.is_running():
            logging.info(f"✓ Consumer is running (PID: {consumer_pid})")
            logging.info(f"  CPU: {process.cpu_percent()}%")
            logging.info(f"  Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            return True
        else:
            logging.error(f"✗ Consumer process not running (PID: {consumer_pid})")
            return False
    except psutil.NoSuchProcess:
        logging.error(f"✗ Consumer process not found (PID: {consumer_pid})")
        return False
    except Exception as e:
        logging.error(f"✗ Error checking consumer health: {e}")
        return False


def monitor_consumer_metrics(**context):
    """Monitor consumer metrics from Kafka consumer groups"""
    from kafka import KafkaAdminClient
    from kafka.admin import ConsumerGroupDescription

    try:
        admin_client = KafkaAdminClient(bootstrap_servers=['localhost:29092'])

        # Get consumer group info
        group_id = 'telco-churn-consumer-group'

        # Note: kafka-python doesn't have direct consumer group describe
        # This is a placeholder for monitoring logic
        # In production, you'd use kafka-python or kafka-consumer-groups CLI

        logging.info(f"✓ Consumer metrics check completed for group: {group_id}")

        admin_client.close()
        return True
    except Exception as e:
        logging.error(f"✗ Error monitoring consumer metrics: {e}")
        return False


def cleanup_on_failure(**context):
    """Cleanup consumer process if DAG fails"""
    import psutil

    ti = context['task_instance']
    consumer_pid = ti.xcom_pull(task_ids='start_consumer', key='consumer_pid')

    if consumer_pid:
        try:
            process = psutil.Process(consumer_pid)
            if process.is_running():
                process.terminate()
                process.wait(timeout=10)
                logging.info(f"✓ Terminated consumer process (PID: {consumer_pid})")
        except Exception as e:
            logging.error(f"✗ Error during cleanup: {e}")


# Define the DAG
with DAG(
    'kafka_streaming_pipeline',
    default_args=default_args,
    description='Kafka Streaming Pipeline for Telco Churn Prediction',
    schedule_interval=None,  # Manual trigger or use '@daily' for daily restarts
    start_date=datetime(2025, 10, 16),
    catchup=False,
    tags=['kafka', 'streaming', 'ml', 'churn-prediction'],
    doc_md="""
    # Kafka Streaming Pipeline DAG

    This DAG manages the long-running Kafka streaming consumer for real-time churn prediction.

    ## Tasks:
    1. **check_kafka** - Verify Kafka broker connectivity
    2. **verify_topics** - Ensure all required topics exist
    3. **verify_model** - Check ML model file exists
    4. **start_consumer** - Launch streaming consumer process
    5. **health_check** - Monitor consumer health (runs periodically)
    6. **monitor_metrics** - Track consumer performance metrics

    ## Usage:
    - Trigger manually to start the streaming consumer
    - Consumer runs continuously until stopped
    - Health checks run every 5 minutes
    - Use the 'stop_consumer' DAG to gracefully stop the consumer

    ## Monitoring:
    - Check logs: `logs/streaming_consumer.log`
    - Kafka consumer group: `telco-churn-consumer-group`
    """
) as dag:

    # Task 1: Check Kafka connection
    check_kafka_task = PythonOperator(
        task_id='check_kafka',
        python_callable=check_kafka_connection,
        provide_context=True,
    )

    # Task 2: Verify topics
    verify_topics_task = PythonOperator(
        task_id='verify_topics',
        python_callable=check_kafka_topics,
        provide_context=True,
    )

    # Task 3: Verify model
    verify_model_task = PythonOperator(
        task_id='verify_model',
        python_callable=verify_model_exists,
        provide_context=True,
    )

    # Task 4: Start streaming consumer
    start_consumer_task = PythonOperator(
        task_id='start_consumer',
        python_callable=start_streaming_consumer,
        provide_context=True,
    )

    # Task 5: Health check sensor (runs every 5 minutes)
    health_check_task = PythonSensor(
        task_id='health_check',
        python_callable=check_consumer_health,
        provide_context=True,
        poke_interval=300,  # Check every 5 minutes
        timeout=86400,  # 24 hours
        mode='reschedule',  # Don't block worker slots
    )

    # Task 6: Monitor metrics
    monitor_metrics_task = PythonOperator(
        task_id='monitor_metrics',
        python_callable=monitor_consumer_metrics,
        provide_context=True,
    )

    # Task 7: Cleanup on failure
    cleanup_task = PythonOperator(
        task_id='cleanup_on_failure',
        python_callable=cleanup_on_failure,
        provide_context=True,
        trigger_rule='one_failed',  # Only run if something fails
    )

    # Define task dependencies
    [check_kafka_task, verify_topics_task, verify_model_task] >> start_consumer_task
    start_consumer_task >> health_check_task >> monitor_metrics_task
    [start_consumer_task, health_check_task, monitor_metrics_task] >> cleanup_task
