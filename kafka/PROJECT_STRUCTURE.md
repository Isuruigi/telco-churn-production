# Kafka Integration - Project Structure

Complete directory structure for the Telco Churn Prediction Kafka integration.

---

## 📁 Directory Structure

```
Telco churn project 1/
│
├── kafka/                              ⭐ KAFKA INTEGRATION (Mini Project 2)
│   │
│   ├── config.py                       # Configuration (Kafka, model, data paths)
│   ├── producer.py                     # Kafka producer (streaming & batch)
│   ├── consumer.py                     # Kafka consumer with ML predictions
│   ├── test_pipeline.py                # Full pipeline testing script
│   ├── requirements-kafka.txt          # Kafka-specific dependencies
│   │
│   ├── README.md                       # Main Kafka integration documentation
│   │
│   ├── airflow/                        # Airflow orchestration
│   │   └── dags/
│   │       ├── kafka_streaming_dag.py  # Streaming pipeline DAG
│   │       └── kafka_batch_dag.py      # Batch pipeline DAG
│   │
│   ├── logs/                           # Application logs
│   │   ├── batch/                      # Batch run logs
│   │   │   ├── producer_*.log
│   │   │   └── consumer_*.log
│   │   └── streaming/                  # Streaming logs
│   │       └── streaming_consumer.log
│   │
│   ├── reports/                        # Generated reports
│   │   └── batch/                      # Batch summary reports
│   │       └── batch_report_*.md
│   │
│   ├── checkpoints/                    # Batch processing checkpoints
│   │   └── producer_checkpoint_*.txt
│   │
│   ├── screenshots/                    # Documentation screenshots
│   │   ├── SCREENSHOTS_GUIDE.md        # Screenshot capture guide
│   │   ├── kafka_logs/                 # Kafka operation screenshots
│   │   │   ├── producer_streaming.png
│   │   │   ├── consumer_streaming.png
│   │   │   ├── topic_messages.png
│   │   │   └── ...
│   │   └── airflow_logs/               # Airflow DAG screenshots
│   │       ├── streaming_dag_graph.png
│   │       ├── batch_dag_graph.png
│   │       └── ...
│   │
│   └── docs/                           # Additional documentation
│       ├── KAFKA_README.md             # Detailed Kafka documentation
│       ├── QUICKSTART.md               # Quick start guide
│       ├── TEST_RESULTS.md             # Validation report
│       └── AIRFLOW_SETUP_GUIDE.md      # Airflow setup guide
│
├── telco-churn-production/            # Mini Project 1 (Original ML project)
│   ├── src/
│   │   ├── models/
│   │   │   └── sklearn_pipeline.pkl   # Trained ML model (used by consumer)
│   │   ├── preprocessing/
│   │   ├── modeling/
│   │   └── ...
│   ├── data/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (used by producer)
│   ├── notebooks/
│   └── ...
│
├── requirements.txt                    # Full project dependencies
├── docker-compose.yml                  # Kafka/Zookeeper Docker setup
└── README.md                           # Main project README

```

---

## 🔧 File Descriptions

### Core Kafka Files

#### **config.py**
- Central configuration for Kafka pipeline
- Kafka broker settings
- Topic names
- Model and data paths (points to telco-churn-production/)
- Streaming and batch configuration
- Logging setup

#### **producer.py**
- TelcoProducer class
- **Streaming mode**: Random sampling at configurable rate
- **Batch mode**: Sequential processing with checkpoints
- Sends messages to `telco-raw-customers` topic
- JSON serialization with timestamps
- Error handling to dead letter queue

#### **consumer.py**
- TelcoConsumer class
- Loads sklearn_pipeline.pkl model
- Consumes from `telco-raw-customers` topic
- **Streaming mode**: Real-time continuous processing
- **Batch mode**: Windowed processing with summaries
- Runs ML predictions
- Publishes to `telco-churn-predictions` topic
- Error handling to dead letter queue

#### **test_pipeline.py**
- Full pipeline testing
- Prerequisites verification
- Multi-threaded producer/consumer execution
- Health checks
- Prediction verification

---

### Airflow DAGs

#### **kafka_streaming_dag.py**
Location: `kafka/airflow/dags/`

**Purpose**: Manage long-running streaming consumer

**Tasks**:
1. Check Kafka connectivity
2. Verify topics exist
3. Verify model file
4. Start streaming consumer
5. Health check (every 5 minutes)
6. Monitor metrics
7. Cleanup on failure

**Paths**:
- KAFKA_DIR: `kafka/`
- PROJECT_ROOT: `Telco churn project 1/`
- Consumer script: `kafka/consumer.py`
- Logs: `kafka/logs/streaming_consumer.log`
- Model: `telco-churn-production/src/models/sklearn_pipeline.pkl`

#### **kafka_batch_dag.py**
Location: `kafka/airflow/dags/`

**Purpose**: Hourly batch processing with reports

**Schedule**: `0 * * * *` (every hour)

**Tasks**:
1. Check prerequisites
2. Run batch producer
3. Run batch consumer
4. Parse summary statistics
5. Generate markdown report
6. Check success threshold
7. Send notification
8. Cleanup

**Paths**:
- KAFKA_DIR: `kafka/`
- Producer script: `kafka/producer.py`
- Consumer script: `kafka/consumer.py`
- Logs: `kafka/logs/batch/`
- Reports: `kafka/reports/batch/`
- Model: `telco-churn-production/src/models/sklearn_pipeline.pkl`
- Data: `telco-churn-production/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

### Documentation Files

#### **README.md** (kafka/)
Main Kafka integration documentation
- Overview and architecture
- Features and capabilities
- Installation instructions
- Usage examples
- Airflow DAG documentation
- Monitoring and troubleshooting

#### **docs/KAFKA_README.md**
Detailed Kafka-specific documentation
- Message schemas
- Topic management
- Consumer groups
- Performance tuning

#### **docs/QUICKSTART.md**
Quick start guide
- 3 ways to run the pipeline
- Common commands
- Example sessions

#### **docs/TEST_RESULTS.md**
Test validation report
- Test execution results
- Performance metrics
- Success criteria

#### **docs/AIRFLOW_SETUP_GUIDE.md**
Complete Airflow setup guide
- Installation steps
- DAG deployment
- Configuration
- Monitoring
- Troubleshooting

#### **screenshots/SCREENSHOTS_GUIDE.md**
Screenshot capture instructions
- 16 required screenshots
- Commands to run
- What to capture
- Naming conventions

---

## 🔀 Path References

### From Kafka Scripts

**config.py:**
```python
BASE_DIR = Path(__file__).parent          # kafka/
PROJECT_ROOT = BASE_DIR.parent            # Telco churn project 1/
MODEL_PATH = PROJECT_ROOT / 'telco-churn-production' / 'src' / 'models' / 'sklearn_pipeline.pkl'
DATA_PATH = PROJECT_ROOT / 'telco-churn-production' / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
```

**producer.py, consumer.py:**
```python
import config  # Imports from kafka/config.py
```

### From Airflow DAGs

**Both DAGs:**
```python
KAFKA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Result: kafka/

PROJECT_ROOT = os.path.abspath(os.path.join(KAFKA_DIR, '..'))
# Result: Telco churn project 1/

sys.path.insert(0, KAFKA_DIR)
# Can now import config, etc.
```

---

## 🚀 Running from Different Locations

### From `kafka/` directory:

```bash
cd kafka

# Run producer
python producer.py --mode streaming

# Run consumer
python consumer.py --mode streaming

# Test pipeline
python test_pipeline.py --verify-only
```

### From project root:

```bash
cd "Telco churn project 1"

# Run with full paths
python kafka/producer.py --mode streaming
python kafka/consumer.py --mode streaming
python kafka/test_pipeline.py --verify-only
```

### Airflow (from anywhere):

```bash
# Copy DAGs
cp kafka/airflow/dags/*.py $AIRFLOW_HOME/dags/

# Trigger from CLI
airflow dags trigger kafka_streaming_pipeline
airflow dags trigger kafka_batch_pipeline
```

---

## 📊 Data Flow

```
telco-churn-production/data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
            ↓
        (read by)
            ↓
    kafka/producer.py
            ↓
        (sends to)
            ↓
    Kafka Topic: telco-raw-customers
            ↓
        (consumed by)
            ↓
    kafka/consumer.py
            ↓
        (loads model)
            ↓
    telco-churn-production/src/models/sklearn_pipeline.pkl
            ↓
        (generates predictions)
            ↓
    Kafka Topic: telco-churn-predictions
            ↓
        (logged to)
            ↓
    kafka/logs/batch/ or kafka/logs/streaming/
```

---

## ✅ Installation Checklist

- [ ] Kafka running on localhost:29092
- [ ] Topics created (or auto-create enabled)
- [ ] Model file exists: `telco-churn-production/src/models/sklearn_pipeline.pkl`
- [ ] Data file exists: `telco-churn-production/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- [ ] Python dependencies installed: `pip install -r requirements.txt`
- [ ] Directory structure created: `logs/`, `reports/`, `checkpoints/`
- [ ] Airflow installed (optional): `pip install apache-airflow==2.7.0`
- [ ] DAGs deployed (optional): Copied to `$AIRFLOW_HOME/dags/`

---

## 📝 Notes

1. **All Kafka files are self-contained in `kafka/` directory**
2. **Airflow DAGs correctly reference kafka/ paths**
3. **Model and data remain in original `telco-churn-production/` location**
4. **Logs, reports, checkpoints stored in `kafka/` subdirectories**
5. **Screenshots documented with capture instructions**
6. **Documentation organized in `kafka/docs/`**

---

*Last Updated: October 16, 2025*
