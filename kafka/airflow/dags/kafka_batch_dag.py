"""
Airflow DAG for Kafka Batch Pipeline - Telco Churn Prediction
Runs hourly: producer → consumer → summary report
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
import os
import sys
import subprocess
import logging
import json
from pathlib import Path

# Add kafka directory to path
KAFKA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(KAFKA_DIR, '..'))
sys.path.insert(0, KAFKA_DIR)

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['your-email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
}


def check_prerequisites(**context):
    """Check Kafka, topics, and model before starting"""
    from kafka import KafkaAdminClient

    errors = []

    # Check Kafka
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=['localhost:29092'],
            request_timeout_ms=5000
        )
        logging.info("✓ Kafka connection successful")

        # Check topics
        required_topics = ['telco-raw-customers', 'telco-churn-predictions', 'telco-deadletter']
        existing_topics = admin_client.list_topics()
        missing_topics = [t for t in required_topics if t not in existing_topics]

        if missing_topics:
            errors.append(f"Missing topics: {missing_topics}")
        else:
            logging.info(f"✓ All topics exist: {required_topics}")

        admin_client.close()
    except Exception as e:
        errors.append(f"Kafka error: {e}")

    # Check model
    model_path = Path(PROJECT_ROOT) / 'telco-churn-production' / 'src' / 'models' / 'sklearn_pipeline.pkl'
    if not model_path.exists():
        errors.append(f"Model not found: {model_path}")
    else:
        logging.info(f"✓ Model exists: {model_path}")

    # Check data
    data_path = Path(PROJECT_ROOT) / 'telco-churn-production' / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if not data_path.exists():
        errors.append(f"Data not found: {data_path}")
    else:
        logging.info(f"✓ Data exists: {data_path}")

    if errors:
        error_msg = "\n".join(errors)
        logging.error(f"Prerequisites check failed:\n{error_msg}")
        raise Exception(error_msg)

    return True


def run_batch_producer(**context):
    """Run producer in batch mode"""
    import subprocess

    execution_date = context['execution_date']
    batch_size = context['params'].get('batch_size', 100)

    producer_script = Path(KAFKA_DIR) / 'producer.py'
    log_dir = Path(KAFKA_DIR) / 'logs' / 'batch'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"producer_{execution_date.strftime('%Y%m%d_%H%M%S')}.log"

    cmd = [
        'python',
        str(producer_script),
        '--mode', 'batch',
        '--batch-size', str(batch_size),
        '--checkpoint-file', f'producer_checkpoint_{execution_date.strftime("%Y%m%d_%H")}.txt'
    ]

    logging.info(f"Starting batch producer: {' '.join(cmd)}")

    with open(log_file, 'w') as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=KAFKA_DIR,
            timeout=3600  # 1 hour timeout
        )

    if result.returncode != 0:
        raise Exception(f"Producer failed with return code {result.returncode}")

    logging.info(f"✓ Producer completed successfully")
    logging.info(f"  Log: {log_file}")

    # Store log file path in XCom
    context['task_instance'].xcom_push(key='producer_log', value=str(log_file))

    return str(log_file)


def run_batch_consumer(**context):
    """Run consumer in batch mode"""
    import subprocess
    import time

    execution_date = context['execution_date']
    window_size = context['params'].get('window_size', 100)
    num_windows = context['params'].get('num_windows', 5)

    # Wait a bit for messages to be produced
    time.sleep(5)

    consumer_script = Path(KAFKA_DIR) / 'consumer.py'
    log_dir = Path(KAFKA_DIR) / 'logs' / 'batch'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"consumer_{execution_date.strftime('%Y%m%d_%H%M%S')}.log"

    cmd = [
        'python',
        str(consumer_script),
        '--mode', 'batch',
        '--window-size', str(window_size),
        '--num-windows', str(num_windows)
    ]

    logging.info(f"Starting batch consumer: {' '.join(cmd)}")

    with open(log_file, 'w') as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=KAFKA_DIR,
            timeout=3600  # 1 hour timeout
        )

    if result.returncode != 0:
        raise Exception(f"Consumer failed with return code {result.returncode}")

    logging.info(f"✓ Consumer completed successfully")
    logging.info(f"  Log: {log_file}")

    # Store log file path in XCom
    context['task_instance'].xcom_push(key='consumer_log', value=str(log_file))

    return str(log_file)


def parse_consumer_summary(**context):
    """Parse consumer log to extract summary statistics"""
    ti = context['task_instance']
    consumer_log = ti.xcom_pull(task_ids='run_consumer', key='consumer_log')

    if not consumer_log or not Path(consumer_log).exists():
        logging.warning("Consumer log not found, skipping summary")
        return {}

    summary = {
        'total_windows': 0,
        'total_messages': 0,
        'total_predictions': 0,
        'total_errors': 0,
        'success_rate': 0.0
    }

    try:
        with open(consumer_log, 'r') as f:
            log_content = f.read()

            # Parse summary section
            if 'Batch Processing Summary:' in log_content:
                lines = log_content.split('\n')
                for line in lines:
                    if 'Total windows processed:' in line:
                        summary['total_windows'] = int(line.split(':')[-1].strip())
                    elif 'Total messages:' in line:
                        summary['total_messages'] = int(line.split(':')[-1].strip())
                    elif 'Total predictions:' in line:
                        summary['total_predictions'] = int(line.split(':')[-1].strip())
                    elif 'Total errors:' in line:
                        summary['total_errors'] = int(line.split(':')[-1].strip())
                    elif 'Success rate:' in line:
                        rate_str = line.split(':')[-1].strip().replace('%', '')
                        summary['success_rate'] = float(rate_str)

        logging.info(f"✓ Parsed summary: {summary}")

        # Store summary in XCom
        ti.xcom_push(key='batch_summary', value=summary)

        return summary
    except Exception as e:
        logging.error(f"Error parsing summary: {e}")
        return summary


def generate_batch_report(**context):
    """Generate summary report for batch run"""
    ti = context['task_instance']
    execution_date = context['execution_date']

    summary = ti.xcom_pull(task_ids='parse_summary', key='batch_summary') or {}
    producer_log = ti.xcom_pull(task_ids='run_producer', key='producer_log')
    consumer_log = ti.xcom_pull(task_ids='run_consumer', key='consumer_log')

    report_dir = Path(KAFKA_DIR) / 'reports' / 'batch'
    report_dir.mkdir(parents=True, exist_ok=True)

    report_file = report_dir / f"batch_report_{execution_date.strftime('%Y%m%d_%H%M%S')}.md"

    report_content = f"""# Kafka Batch Pipeline Report

**Execution Date:** {execution_date.strftime('%Y-%m-%d %H:%M:%S')}
**DAG Run ID:** {context['dag_run'].run_id}

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Windows Processed | {summary.get('total_windows', 'N/A')} |
| Total Messages | {summary.get('total_messages', 'N/A')} |
| Total Predictions | {summary.get('total_predictions', 'N/A')} |
| Total Errors | {summary.get('total_errors', 'N/A')} |
| Success Rate | {summary.get('success_rate', 'N/A')}% |

---

## Pipeline Status

✅ **Status:** {'SUCCESS' if summary.get('success_rate', 0) > 95 else 'WARNING'}

---

## Log Files

- **Producer Log:** `{producer_log or 'N/A'}`
- **Consumer Log:** `{consumer_log or 'N/A'}`
- **Report:** `{report_file}`

---

## Predictions Output

Predictions published to Kafka topic: `telco-churn-predictions`

To view predictions:
```bash
kafka-console-consumer --bootstrap-server localhost:29092 \\
  --topic telco-churn-predictions \\
  --from-beginning
```

---

## Next Steps

1. Review prediction results in `telco-churn-predictions` topic
2. Check for any messages in dead letter queue: `telco-deadletter`
3. Analyze churn probability distribution
4. Generate business insights

---

*Generated by Airflow Batch Pipeline DAG*
"""

    with open(report_file, 'w') as f:
        f.write(report_content)

    logging.info(f"✓ Generated batch report: {report_file}")

    # Store report path in XCom
    ti.xcom_push(key='report_file', value=str(report_file))

    return str(report_file)


def check_success_threshold(**context):
    """Check if success rate meets threshold, decide next task"""
    ti = context['task_instance']
    summary = ti.xcom_pull(task_ids='parse_summary', key='batch_summary') or {}

    success_rate = summary.get('success_rate', 0)
    threshold = 95.0

    if success_rate >= threshold:
        logging.info(f"✓ Success rate {success_rate}% meets threshold {threshold}%")
        return 'send_success_notification'
    else:
        logging.warning(f"✗ Success rate {success_rate}% below threshold {threshold}%")
        return 'send_failure_alert'


def send_success_notification(**context):
    """Log success notification"""
    ti = context['task_instance']
    summary = ti.xcom_pull(task_ids='parse_summary', key='batch_summary') or {}

    logging.info("=" * 60)
    logging.info("BATCH PIPELINE COMPLETED SUCCESSFULLY")
    logging.info(f"Messages: {summary.get('total_messages', 'N/A')}")
    logging.info(f"Predictions: {summary.get('total_predictions', 'N/A')}")
    logging.info(f"Success Rate: {summary.get('success_rate', 'N/A')}%")
    logging.info("=" * 60)


def send_failure_alert(**context):
    """Log failure alert"""
    ti = context['task_instance']
    summary = ti.xcom_pull(task_ids='parse_summary', key='batch_summary') or {}

    logging.error("=" * 60)
    logging.error("BATCH PIPELINE COMPLETED WITH WARNINGS")
    logging.error(f"Messages: {summary.get('total_messages', 'N/A')}")
    logging.error(f"Errors: {summary.get('total_errors', 'N/A')}")
    logging.error(f"Success Rate: {summary.get('success_rate', 'N/A')}%")
    logging.error("=" * 60)


# Define the DAG
with DAG(
    'kafka_batch_pipeline',
    default_args=default_args,
    description='Hourly Kafka Batch Pipeline for Telco Churn Prediction',
    schedule_interval='0 * * * *',  # Run every hour at :00
    start_date=datetime(2025, 10, 16),
    catchup=False,
    max_active_runs=1,  # Only one batch run at a time
    params={
        'batch_size': 100,
        'window_size': 100,
        'num_windows': 5
    },
    tags=['kafka', 'batch', 'ml', 'churn-prediction', 'hourly'],
    doc_md="""
    # Kafka Batch Pipeline DAG

    Runs hourly to process batches of customer data for churn prediction.

    ## Workflow:
    1. **Check Prerequisites** - Verify Kafka, topics, model, and data
    2. **Run Producer** - Send batch of customer data to Kafka
    3. **Run Consumer** - Process messages and generate predictions
    4. **Parse Summary** - Extract metrics from consumer logs
    5. **Generate Report** - Create markdown summary report
    6. **Check Threshold** - Evaluate success rate
    7. **Notify** - Send success/failure notification

    ## Schedule:
    - Runs hourly at :00 (e.g., 10:00, 11:00, 12:00, ...)
    - Only one run active at a time

    ## Parameters:
    - `batch_size`: Number of records producer sends (default: 100)
    - `window_size`: Consumer window size (default: 100)
    - `num_windows`: Number of windows to process (default: 5)

    ## Outputs:
    - Predictions: `telco-churn-predictions` Kafka topic
    - Logs: `logs/batch/producer_*.log`, `logs/batch/consumer_*.log`
    - Reports: `reports/batch/batch_report_*.md`

    ## Monitoring:
    - Check DAG run status in Airflow UI
    - Review batch reports in `reports/batch/`
    - Monitor Kafka consumer lag
    """
) as dag:

    # Task 1: Check prerequisites
    check_prerequisites_task = PythonOperator(
        task_id='check_prerequisites',
        python_callable=check_prerequisites,
        provide_context=True,
    )

    # Task 2: Run batch producer
    run_producer_task = PythonOperator(
        task_id='run_producer',
        python_callable=run_batch_producer,
        provide_context=True,
    )

    # Task 3: Run batch consumer
    run_consumer_task = PythonOperator(
        task_id='run_consumer',
        python_callable=run_batch_consumer,
        provide_context=True,
    )

    # Task 4: Parse consumer summary
    parse_summary_task = PythonOperator(
        task_id='parse_summary',
        python_callable=parse_consumer_summary,
        provide_context=True,
    )

    # Task 5: Generate batch report
    generate_report_task = PythonOperator(
        task_id='generate_report',
        python_callable=generate_batch_report,
        provide_context=True,
    )

    # Task 6: Check success threshold (branching)
    check_threshold_task = BranchPythonOperator(
        task_id='check_threshold',
        python_callable=check_success_threshold,
        provide_context=True,
    )

    # Task 7a: Success notification
    success_notification_task = PythonOperator(
        task_id='send_success_notification',
        python_callable=send_success_notification,
        provide_context=True,
    )

    # Task 7b: Failure alert
    failure_alert_task = PythonOperator(
        task_id='send_failure_alert',
        python_callable=send_failure_alert,
        provide_context=True,
    )

    # Task 8: Cleanup (always runs)
    cleanup_task = BashOperator(
        task_id='cleanup',
        bash_command='echo "Batch pipeline completed at $(date)"',
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Define task dependencies
    check_prerequisites_task >> run_producer_task >> run_consumer_task
    run_consumer_task >> parse_summary_task >> generate_report_task
    generate_report_task >> check_threshold_task
    check_threshold_task >> [success_notification_task, failure_alert_task]
    [success_notification_task, failure_alert_task] >> cleanup_task
