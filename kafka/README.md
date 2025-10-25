# Telco Churn Prediction - Kafka Integration (Mini Project 2)

Complete real-time and batch churn prediction system using Apache Kafka and Apache Airflow.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Kafka](https://img.shields.io/badge/Kafka-2.6-orange.svg)](https://kafka.apache.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.7-green.svg)](https://airflow.apache.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Airflow DAGs](#airflow-dags)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Screenshots](#screenshots)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project implements a production-ready Kafka-based pipeline for real-time telco customer churn prediction with Airflow orchestration.

**Key Components:**
- **Real-time Streaming**: Continuous processing with sub-second latency
- **Batch Processing**: Hourly scheduled predictions
- **Airflow Orchestration**: Automated DAG management
- **ML Integration**: sklearn pipeline for churn prediction
- **Error Handling**: Dead letter queue for failed messages

---

## 🏗️ Architecture

```
CSV Data → Producer → Kafka → Consumer + ML Model → Predictions Topic
                                    ↓
                            Dead Letter Queue
                                    ↓
                            Logs & Reports
```

**Airflow orchestrates the entire pipeline with health checks and automated reporting.**

---

## ✨ Features

### Producer (`producer.py`)
- ✅ Streaming mode: Configurable event rate (events/sec)
- ✅ Batch mode: Checkpointed batch processing
- ✅ JSON serialization with timestamps
- ✅ Error handling with dead letter queue

### Consumer (`consumer.py`)
- ✅ Real-time ML predictions
- ✅ Batch windowed processing
- ✅ Churn probability scoring
- ✅ Summary statistics generation

### Airflow DAGs
- ✅ **Streaming DAG**: Long-running consumer with health checks
- ✅ **Batch DAG**: Hourly scheduled batch processing
- ✅ Automated report generation
- ✅ Success/failure notifications

---

## 📁 Project Structure

```
├── config.py                       # Configuration
├── producer.py                     # Kafka producer
├── consumer.py                     # Kafka consumer + ML
├── test_pipeline.py                # Testing script
├── requirements.txt                # Dependencies
│
├── airflow/dags/
│   ├── kafka_streaming_dag.py     # Streaming DAG
│   └── kafka_batch_dag.py         # Batch DAG
│
├── logs/                           # Application logs
│   ├── batch/                      # Batch run logs
│   └── streaming/                  # Streaming logs
│
├── reports/batch/                  # Generated reports
├── screenshots/                    # Documentation screenshots
│   ├── kafka_logs/                 # Kafka screenshots
│   └── airflow_logs/               # Airflow screenshots
│
└── checkpoints/                    # Batch checkpoints
```

---

## 🔧 Prerequisites

**Required:**
- Python 3.8+
- Apache Kafka 2.6+ (localhost:29092)
- Trained model: `telco-churn-production/src/models/sklearn_pipeline.pkl`
- Dataset: `telco-churn-production/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Optional:**
- Apache Airflow 2.7+ (for DAG orchestration)
- Docker (for containerized Kafka)

---

## 📦 Installation

### 1. Install Dependencies

```bash
# Full installation
pip install -r requirements.txt

# Kafka-only (minimal)
pip install -r requirements-kafka.txt
```

### 2. Start Kafka

```bash
# Using Docker Compose
docker-compose up -d

# Verify
docker ps
```

### 3. Create Kafka Topics

```bash
# Auto-create with test script
python test_pipeline.py --verify-only

# Or manually
kafka-topics --create --topic telco-raw-customers --bootstrap-server localhost:29092 --partitions 1 --replication-factor 1
kafka-topics --create --topic telco-churn-predictions --bootstrap-server localhost:29092 --partitions 1 --replication-factor 1
kafka-topics --create --topic telco-deadletter --bootstrap-server localhost:29092 --partitions 1 --replication-factor 1
```

### 4. Verify Setup

```bash
python test_pipeline.py --verify-only
```

Expected:
```
✓ Kafka connection verified
✓ All required topics exist
✓ Model file found
✓ CSV file found
```

---

## 🚀 Quick Start

### Option 1: Full Pipeline Test

```bash
# Streaming test (20 seconds)
python test_pipeline.py --mode streaming --duration 20

# Batch test
python test_pipeline.py --mode batch
```

### Option 2: Manual Execution

```bash
# Terminal 1: Consumer
python consumer.py --mode streaming

# Terminal 2: Producer
python producer.py --mode streaming --events-per-sec 10 --duration 60
```

---

## 📖 Usage

### Producer Commands

**Streaming:**
```bash
# Default: 10 events/sec, continuous
python producer.py --mode streaming

# Custom: 50 events/sec for 2 minutes
python producer.py --mode streaming --events-per-sec 50 --duration 120
```

**Batch:**
```bash
# Default: 100 records per batch
python producer.py --mode batch

# Large batch: 1000 records
python producer.py --mode batch --batch-size 1000

# Resume from checkpoint
python producer.py --mode batch --resume
```

### Consumer Commands

**Streaming:**
```bash
# Continuous
python consumer.py --mode streaming

# 5 minutes
python consumer.py --mode streaming --duration 300
```

**Batch:**
```bash
# Default: 100 messages per window
python consumer.py --mode batch

# Custom: 50 messages, 10 windows
python consumer.py --mode batch --window-size 50 --num-windows 10
```

---

## 🔄 Airflow DAGs

### Setup Airflow

```bash
# Initialize
export AIRFLOW_HOME=~/airflow
airflow db init

# Create admin user
airflow users create --username admin --password admin --role Admin --email admin@example.com

# Copy DAGs
cp airflow/dags/*.py $AIRFLOW_HOME/dags/

# Start services
airflow webserver --port 8080  # Terminal 1
airflow scheduler               # Terminal 2
```

Access UI: http://localhost:8080

### Streaming Pipeline DAG

**Purpose**: Manage long-running consumer

**Tasks**:
1. Check Kafka connectivity
2. Verify topics
3. Verify model
4. Start streaming consumer
5. Health check (every 5 min)
6. Monitor metrics

**Usage**: Trigger manually from Airflow UI

### Batch Pipeline DAG

**Purpose**: Hourly batch processing

**Schedule**: Every hour at :00

**Tasks**:
1. Check prerequisites
2. Run producer (batch mode)
3. Run consumer (batch mode)
4. Parse summary
5. Generate report
6. Check success threshold
7. Send notification

**Parameters**:
- `batch_size`: 100 (default)
- `window_size`: 100
- `num_windows`: 5

**Outputs**:
- Logs: `logs/batch/producer_*.log`, `logs/batch/consumer_*.log`
- Reports: `reports/batch/batch_report_*.md`

---

## ⚙️ Configuration

Edit `config.py`:

```python
# Kafka servers
KAFKA_BOOTSTRAP_SERVERS = ['localhost:29092']

# Topics
KAFKA_TOPICS = {
    'raw_customers': 'telco-raw-customers',
    'predictions': 'telco-churn-predictions',
    'deadletter': 'telco-deadletter'
}

# Streaming
STREAMING_CONFIG = {
    'default_events_per_sec': 10
}

# Batch
BATCH_CONFIG = {
    'default_batch_size': 100
}
```

---

## 📊 Monitoring

### View Kafka Messages

```bash
# Raw customer data
kafka-console-consumer --bootstrap-server localhost:29092 --topic telco-raw-customers --from-beginning

# Predictions
kafka-console-consumer --bootstrap-server localhost:29092 --topic telco-churn-predictions --from-beginning

# Errors
kafka-console-consumer --bootstrap-server localhost:29092 --topic telco-deadletter --from-beginning
```

### Consumer Group Status

```bash
# List groups
kafka-consumer-groups --bootstrap-server localhost:29092 --list

# Describe group
kafka-consumer-groups --bootstrap-server localhost:29092 --group telco-churn-consumer-group --describe
```

### Application Logs

```bash
# Streaming
tail -f logs/streaming_consumer.log

# Batch
ls -lh logs/batch/
cat logs/batch/consumer_20251016_120000.log
```

### Airflow Monitoring

- Web UI: http://localhost:8080
- DAG Runs: Execution history
- Task Logs: Individual outputs
- Gantt Chart: Task durations

---

## 📸 Screenshots

### Kafka Logs (`screenshots/kafka_logs/`)
- `producer_streaming.png` - Producer output
- `consumer_predictions.png` - Consumer predictions
- `topic_messages.png` - Kafka messages
- `consumer_group.png` - Consumer group status

### Airflow Logs (`screenshots/airflow_logs/`)
- `streaming_dag.png` - Streaming DAG graph
- `batch_dag.png` - Batch DAG graph
- `task_logs.png` - Task execution
- `batch_report.png` - Generated report

**To capture:**
1. Run commands
2. Screenshot terminal/UI
3. Save in appropriate directory
4. Document in this README

---

## 🔧 Troubleshooting

### Kafka Connection Failed

```bash
# Check Kafka running
docker ps
netstat -an | grep 29092

# Test connection
python -c "from kafka import KafkaAdminClient; KafkaAdminClient(bootstrap_servers=['localhost:29092']).close(); print('OK')"
```

### Model Not Found

```bash
# Check path
ls -la telco-churn-production/src/models/sklearn_pipeline.pkl

# Update config.py if needed
```

### Consumer Not Receiving Messages

```bash
# Check offset
kafka-consumer-groups --bootstrap-server localhost:29092 --group telco-churn-consumer-group --describe

# Reset to earliest
kafka-consumer-groups --bootstrap-server localhost:29092 --group telco-churn-consumer-group --reset-offsets --to-earliest --all-topics --execute
```

### Airflow DAG Not Visible

```bash
# Check syntax
python airflow/dags/kafka_batch_dag.py

# List DAGs
airflow dags list

# Check logs
tail -f $AIRFLOW_HOME/logs/scheduler/latest/*.log
```

---

## 📚 Additional Documentation

- **KAFKA_README.md** - Detailed Kafka documentation
- **QUICKSTART.md** - Quick start guide
- **TEST_RESULTS.md** - Validation report
- **requirements.txt** - Full dependency list

---

## 🙏 Acknowledgments

- Apache Kafka - Distributed streaming
- Apache Airflow - Workflow orchestration
- scikit-learn - Machine learning
- Mini Project 1 - Base model

---

**Built for real-time ML predictions** 🚀

*Last Updated: October 25, 2025*
