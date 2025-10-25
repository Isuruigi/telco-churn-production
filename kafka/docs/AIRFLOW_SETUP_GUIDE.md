# Airflow Setup Guide - Kafka Integration

Complete guide to set up and run Apache Airflow DAGs for the Telco Churn Kafka pipeline.

---

## 📋 Overview

This guide covers:
1. Installing Apache Airflow
2. Configuring Airflow for the project
3. Deploying DAGs
4. Running and monitoring workflows
5. Troubleshooting

---

## 🔧 Prerequisites

- Python 3.8+
- Kafka running (localhost:29092)
- Virtual environment activated
- Project dependencies installed (`requirements.txt`)

---

## 📦 Installation

### Step 1: Install Airflow

```bash
# Set Airflow home directory
export AIRFLOW_HOME=~/airflow

# Install Airflow with constraints
AIRFLOW_VERSION=2.7.0
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Install additional providers
pip install apache-airflow-providers-apache-kafka
```

### Step 2: Initialize Airflow Database

```bash
# Initialize the database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### Step 3: Configure Airflow

Edit `~/airflow/airflow.cfg`:

```ini
[core]
# Load examples (set to False for production)
load_examples = False

# Parallelism
parallelism = 32
max_active_runs_per_dag = 1

[webserver]
# Web server port
web_server_port = 8080

# Authentication
authenticate = True

[scheduler]
# Scheduler settings
scheduler_heartbeat_sec = 5
```

---

## 📁 Deploy DAGs

### Copy DAG Files

```bash
# Copy DAGs to Airflow DAGs directory
cp airflow/dags/kafka_streaming_dag.py $AIRFLOW_HOME/dags/
cp airflow/dags/kafka_batch_dag.py $AIRFLOW_HOME/dags/

# Verify DAGs are recognized
airflow dags list | grep kafka
```

Expected output:
```
kafka_batch_pipeline
kafka_streaming_pipeline
```

### Set Project Path

Both DAGs use `PROJECT_ROOT` to locate scripts. Ensure the path is correct:

```python
# In kafka_streaming_dag.py and kafka_batch_dag.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
```

This assumes:
```
C:\Users\IPK\Telco churn project 1\
    airflow/dags/  (DAG files)
    config.py      (Project root)
    producer.py
    consumer.py
```

---

## 🚀 Start Airflow

### Terminal 1: Webserver

```bash
# Start webserver
airflow webserver --port 8080
```

### Terminal 2: Scheduler

```bash
# Start scheduler
airflow scheduler
```

### Access Web UI

Open browser: **http://localhost:8080**

Login:
- Username: `admin`
- Password: `admin` (or what you set)

---

## 📊 Using the DAGs

### Streaming Pipeline DAG

**Purpose:** Manage long-running streaming consumer with health monitoring

#### Enable DAG

1. Go to http://localhost:8080
2. Find `kafka_streaming_pipeline`
3. Toggle switch to **ON**

#### Trigger DAG

**Manual Trigger:**
1. Click on `kafka_streaming_pipeline`
2. Click **Play** button (▶) in top right
3. Confirm trigger

**CLI Trigger:**
```bash
airflow dags trigger kafka_streaming_pipeline
```

#### Monitor Execution

1. **Graph View:** Shows task dependencies
2. **Grid View:** Shows task status over time
3. **Gantt View:** Shows task duration
4. **Task Logs:** Click on task → Logs

#### Expected Behavior

Tasks run in sequence:
1. ✅ `check_kafka` - Verifies Kafka connection
2. ✅ `verify_topics` - Checks topics exist
3. ✅ `verify_model` - Confirms model file
4. ✅ `start_consumer` - Launches streaming consumer
5. 🔄 `health_check` - Monitors consumer (every 5 min)
6. ✅ `monitor_metrics` - Tracks performance

Consumer runs continuously until manually stopped.

#### Stop Streaming Consumer

**Option 1: Stop DAG Run**
1. Click on running DAG instance
2. Click **Stop** button

**Option 2: Kill Process**
```bash
# Find consumer PID
ps aux | grep consumer.py

# Kill process
kill <PID>
```

---

### Batch Pipeline DAG

**Purpose:** Hourly batch processing with automated reporting

#### Enable DAG

1. Go to http://localhost:8080
2. Find `kafka_batch_pipeline`
3. Toggle switch to **ON**

#### Schedule

- **Default:** Every hour at :00 (e.g., 10:00, 11:00, 12:00)
- **Modify:** Edit `schedule_interval` in `kafka_batch_dag.py`

```python
schedule_interval='0 * * * *',  # Hourly at :00
# schedule_interval='*/30 * * * *',  # Every 30 minutes
# schedule_interval='0 9 * * *',  # Daily at 9:00 AM
```

#### Manual Trigger

**Web UI:**
1. Click on `kafka_batch_pipeline`
2. Click **Play** button
3. Optionally override parameters

**CLI:**
```bash
airflow dags trigger kafka_batch_pipeline

# With custom parameters
airflow dags trigger kafka_batch_pipeline \
  --conf '{"batch_size": 200, "window_size": 200, "num_windows": 3}'
```

#### Monitor Execution

**Real-time:**
1. Click on DAG run
2. Watch tasks turn green (success) or red (failure)

**Task Flow:**
1. ✅ `check_prerequisites` - Verify setup
2. ✅ `run_producer` - Send messages to Kafka
3. ✅ `run_consumer` - Process and predict
4. ✅ `parse_summary` - Extract metrics
5. ✅ `generate_report` - Create markdown report
6. 🔀 `check_threshold` - Branch based on success rate
7. ✅/❌ `send_success_notification` OR `send_failure_alert`
8. ✅ `cleanup` - Finalize

#### View Results

**Batch Reports:**
```bash
# List reports
ls -lh reports/batch/

# View latest report
cat reports/batch/batch_report_*.md | tail -100
```

**Task Logs:**
1. Click on task (e.g., `run_consumer`)
2. Click **Log** button
3. View output

**Sample Report:**
```markdown
# Kafka Batch Pipeline Report

**Execution Date:** 2025-10-16 15:00:00

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Messages | 500 |
| Total Predictions | 500 |
| Total Errors | 0 |
| Success Rate | 100.0% |

✅ **Status:** SUCCESS
```

---

## ⚙️ Configuration

### DAG Parameters

**Streaming DAG:**
- No configurable parameters (runs continuously)

**Batch DAG:**

Edit `params` in `kafka_batch_dag.py`:

```python
params={
    'batch_size': 100,      # Producer batch size
    'window_size': 100,     # Consumer window size
    'num_windows': 5        # Number of windows
}
```

Or override at runtime:
```bash
airflow dags trigger kafka_batch_pipeline \
  --conf '{"batch_size": 500}'
```

### Email Notifications

Edit `default_args` in DAG files:

```python
default_args = {
    'email': ['your-email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
}
```

Configure SMTP in `airflow.cfg`:
```ini
[smtp]
smtp_host = smtp.gmail.com
smtp_starttls = True
smtp_ssl = False
smtp_user = your-email@example.com
smtp_password = your-app-password
smtp_port = 587
smtp_mail_from = your-email@example.com
```

---

## 🔍 Monitoring & Logs

### Airflow Logs

**Location:** `$AIRFLOW_HOME/logs/`

**Structure:**
```
logs/
├── dag_id=kafka_batch_pipeline/
│   ├── run_id=manual__2025-10-16T15:00:00+00:00/
│   │   ├── task_id=check_prerequisites/
│   │   │   └── attempt=1.log
│   │   ├── task_id=run_producer/
│   │   └── task_id=run_consumer/
│   └── ...
└── scheduler/
    └── latest/
```

**View Logs:**
```bash
# Scheduler logs
tail -f $AIRFLOW_HOME/logs/scheduler/latest/*.log

# Specific DAG run
tail -f $AIRFLOW_HOME/logs/dag_id=kafka_batch_pipeline/run_id=*/task_id=run_consumer/attempt=1.log
```

### Application Logs

**Producer/Consumer Logs:** `logs/batch/`

```bash
# List batch logs
ls -lh logs/batch/

# View producer log
cat logs/batch/producer_20251016_150000.log

# View consumer log
cat logs/batch/consumer_20251016_150000.log
```

### Kafka Monitoring

```bash
# Consumer group lag
kafka-consumer-groups --bootstrap-server localhost:29092 \
  --group telco-churn-consumer-group --describe
```

---

## 🐛 Troubleshooting

### DAG Not Appearing

**Problem:** DAG doesn't show in UI

**Solutions:**
```bash
# Check DAG file syntax
python $AIRFLOW_HOME/dags/kafka_batch_dag.py

# List all DAGs
airflow dags list

# Parse DAG
airflow dags show kafka_batch_pipeline

# Check scheduler logs
tail -f $AIRFLOW_HOME/logs/scheduler/latest/*.log
```

### Import Errors

**Problem:** `ModuleNotFoundError` in task logs

**Solutions:**
```bash
# Ensure project root in path
# In DAG files, verify:
PROJECT_ROOT = os.path.abspath(...)
sys.path.insert(0, PROJECT_ROOT)

# Install dependencies in Airflow environment
pip install -r requirements.txt
```

### Task Timeout

**Problem:** Task exceeds `execution_timeout`

**Solutions:**
```python
# Increase timeout in default_args
default_args = {
    'execution_timeout': timedelta(hours=2),  # Increase to 2 hours
}
```

### Consumer Not Starting

**Problem:** `start_consumer` task fails

**Solutions:**
```bash
# Check Kafka running
docker ps | grep kafka

# Verify model exists
ls -la telco-churn-production/src/models/sklearn_pipeline.pkl

# Check logs
cat $AIRFLOW_HOME/logs/dag_id=kafka_streaming_pipeline/.../task_id=start_consumer/attempt=1.log
```

### Batch Reports Not Generated

**Problem:** Report file not created

**Solutions:**
```bash
# Check directory permissions
ls -ld reports/batch/

# Create directory manually
mkdir -p reports/batch

# Check generate_report task logs
```

---

## 🔄 Maintenance

### Reset DAG

```bash
# Clear all task instances
airflow tasks clear kafka_batch_pipeline

# Delete DAG runs
airflow dags delete kafka_batch_pipeline
```

### Pause/Unpause DAG

```bash
# Pause
airflow dags pause kafka_batch_pipeline

# Unpause
airflow dags unpause kafka_batch_pipeline
```

### Update DAG

1. Edit DAG file in `airflow/dags/`
2. Save changes
3. Wait ~30 seconds for Airflow to detect changes
4. Refresh web UI

---

## 📊 Performance Tuning

### Increase Parallelism

Edit `airflow.cfg`:
```ini
[core]
parallelism = 64
dag_concurrency = 32
max_active_runs_per_dag = 3
```

### Optimize Task Execution

```python
# Use pools for resource management
from airflow.models import Pool

# In DAG
task = PythonOperator(
    pool='kafka_pool',
    pool_slots=2
)
```

---

## 📚 Additional Resources

- **Airflow Docs:** https://airflow.apache.org/docs/
- **Kafka Integration:** `KAFKA_README.md`
- **Quick Start:** `QUICKSTART.md`
- **Main README:** `README_KAFKA.md`

---

## ✅ Quick Reference

**Start Airflow:**
```bash
airflow webserver --port 8080  # Terminal 1
airflow scheduler               # Terminal 2
```

**Access UI:** http://localhost:8080

**Trigger DAGs:**
```bash
# Streaming
airflow dags trigger kafka_streaming_pipeline

# Batch
airflow dags trigger kafka_batch_pipeline
```

**View Logs:**
```bash
tail -f $AIRFLOW_HOME/logs/scheduler/latest/*.log
```

**Check DAG Status:**
```bash
airflow dags list-runs -d kafka_batch_pipeline
```

---

**Happy Orchestrating! 🚀**

*Last Updated: October 16, 2025*
