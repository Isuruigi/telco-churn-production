# Telco Churn Prediction - Kafka Integration (Mini Project 2)

This directory contains the Kafka integration for real-time and batch churn prediction.

## Project Structure

```
├── config.py                    # Configuration file with Kafka and model settings
├── producer.py                  # Kafka producer (streaming & batch modes)
├── consumer.py                  # Kafka consumer with ML predictions (streaming & batch modes)
├── test_pipeline.py             # Full pipeline testing script
├── checkpoints/                 # Checkpoint files for batch processing
├── models/                      # ML models directory
│   └── churn_model.pkl         # Trained model (symlink to actual model)
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (symlink to actual data)
```

## Prerequisites

### 1. Kafka Setup
Ensure Kafka is running on `localhost:29092` with the following topics:
- `telco-raw-customers` - Raw customer data
- `telco-churn-predictions` - Prediction results
- `telco-deadletter` - Failed messages

To create topics manually:
```bash
kafka-topics --create --topic telco-raw-customers --bootstrap-server localhost:29092 --partitions 1 --replication-factor 1
kafka-topics --create --topic telco-churn-predictions --bootstrap-server localhost:29092 --partitions 1 --replication-factor 1
kafka-topics --create --topic telco-deadletter --bootstrap-server localhost:29092 --partitions 1 --replication-factor 1
```

Or let `test_pipeline.py` create them automatically.

### 2. Python Dependencies
Install required packages:
```bash
pip install kafka-python pandas numpy scikit-learn joblib
```

### 3. Model and Data
The configuration expects:
- Model: `telco-churn-production/src/models/sklearn_pipeline.pkl`
- Data: `telco-churn-production/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Usage

### Testing the Pipeline

Run the full pipeline test (easiest way to get started):
```bash
# Test streaming mode (default)
python test_pipeline.py

# Test batch mode
python test_pipeline.py --mode batch

# Test both modes
python test_pipeline.py --mode both

# Verify prerequisites only
python test_pipeline.py --verify-only

# Custom parameters
python test_pipeline.py --mode streaming --events-per-sec 20 --duration 30
```

### Producer (Standalone)

**Streaming Mode** - Continuously sample and send random customers:
```bash
# Default: 10 events/sec, runs indefinitely
python producer.py --mode streaming

# Custom rate: 50 events/sec for 60 seconds
python producer.py --mode streaming --events-per-sec 50 --duration 60
```

**Batch Mode** - Send entire dataset in chunks:
```bash
# Default: 100 records per batch
python producer.py --mode batch

# Custom batch size
python producer.py --mode batch --batch-size 500

# Resume from checkpoint
python producer.py --mode batch --resume --checkpoint-file my_checkpoint.txt
```

### Consumer (Standalone)

**Streaming Mode** - Continuously consume and predict:
```bash
# Default: runs indefinitely
python consumer.py --mode streaming

# Run for 120 seconds
python consumer.py --mode streaming --duration 120
```

**Batch Mode** - Process in windows:
```bash
# Default: 100 messages per window, process all available
python consumer.py --mode batch

# Custom window size and limit
python consumer.py --mode batch --window-size 200 --num-windows 5
```

### Configuration

Edit `config.py` to customize:
- Kafka bootstrap servers
- Topic names
- Model path
- Batch sizes
- Logging configuration

## Message Schemas

### Raw Customer Message (Producer → telco-raw-customers)
```json
{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85,
  "event_ts": "2025-10-16T13:45:30.123456"
}
```

### Prediction Message (Consumer → telco-churn-predictions)
```json
{
  "customerID": "7590-VHVEG",
  "churn_probability": 0.82,
  "prediction": "Yes",
  "confidence_score": 0.82,
  "event_ts": "2025-10-16T13:45:30.123456",
  "processed_ts": "2025-10-16T13:45:30.456789"
}
```

### Dead Letter Message (Consumer → telco-deadletter)
```json
{
  "original_message": { ... },
  "error": "Error message",
  "stage": "prediction",
  "failed_at": "2025-10-16T13:45:30.789012"
}
```

## Monitoring

### View Messages in Topics

**Raw customers:**
```bash
kafka-console-consumer --bootstrap-server localhost:29092 --topic telco-raw-customers --from-beginning
```

**Predictions:**
```bash
kafka-console-consumer --bootstrap-server localhost:29092 --topic telco-churn-predictions --from-beginning
```

**Dead letter queue:**
```bash
kafka-console-consumer --bootstrap-server localhost:29092 --topic telco-deadletter --from-beginning
```

### Check Consumer Groups
```bash
kafka-consumer-groups --bootstrap-server localhost:29092 --list
kafka-consumer-groups --bootstrap-server localhost:29092 --group telco-churn-consumer-group --describe
```

### Logs
Both producer and consumer log to:
- Console (INFO level)
- File: `kafka_pipeline.log` (DEBUG level)

## Features

### Producer
- ✅ Streaming mode with configurable rate (events/sec)
- ✅ Batch mode with checkpoint/resume support
- ✅ JSON message serialization
- ✅ CustomerID as message key
- ✅ Event timestamps
- ✅ Error handling with dead letter queue
- ✅ Progress logging

### Consumer
- ✅ Streaming mode for real-time predictions
- ✅ Batch mode with windowed processing
- ✅ ML model loading (scikit-learn pipeline)
- ✅ Data preprocessing
- ✅ Churn probability prediction
- ✅ Confidence scores
- ✅ Error handling with dead letter queue
- ✅ Batch summary statistics

### Testing
- ✅ Automated full pipeline testing
- ✅ Prerequisites verification (Kafka, topics, model, data)
- ✅ Multi-threaded producer/consumer execution
- ✅ Prediction verification
- ✅ Support for both streaming and batch modes

## Error Handling

Both producer and consumer implement robust error handling:
1. Failed messages are sent to `telco-deadletter` topic
2. Errors are logged with full context
3. Processing continues even if individual messages fail
4. Batch mode supports checkpointing for resumability

## Performance Tips

1. **Streaming Mode:**
   - Adjust `--events-per-sec` based on your Kafka cluster capacity
   - Monitor consumer lag using consumer group commands
   - Use multiple consumer instances for parallel processing

2. **Batch Mode:**
   - Larger batch sizes reduce overhead but increase memory usage
   - Use checkpoints for long-running batch jobs
   - Adjust `max_poll_records` in config.py for consumer tuning

3. **General:**
   - Monitor `kafka_pipeline.log` for performance insights
   - Check dead letter queue for systematic issues
   - Adjust Kafka topic partitions for higher throughput

## Troubleshooting

**Kafka connection refused:**
- Ensure Kafka is running: `docker ps` or check your Kafka service
- Verify bootstrap server address in `config.py`

**Model not found:**
- Check model path in `config.py`
- Ensure model file exists and is accessible

**Consumer not receiving messages:**
- Check consumer group offset: `kafka-consumer-groups --describe`
- Try resetting offset: `--auto-offset-reset earliest`
- Verify messages in topic with console consumer

**Predictions failing:**
- Check model compatibility with input data
- Review preprocessing in consumer
- Check dead letter queue for error messages

## Next Steps

1. **Monitoring Dashboard:** Create Grafana dashboard for Kafka metrics
2. **Schema Registry:** Implement Avro schemas for better data validation
3. **Scaling:** Deploy multiple consumer instances for parallel processing
4. **Model Versioning:** Add support for loading different model versions
5. **A/B Testing:** Implement shadow predictions with multiple models
6. **Performance Metrics:** Add prediction latency tracking
7. **Data Validation:** Add input data validation before prediction

## License

Part of the Telco Churn Prediction project.
