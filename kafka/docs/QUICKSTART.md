# Quick Start Guide - Kafka Integration

## Step 1: Verify Kafka is Running

Check if Kafka is running on `localhost:29092`:

```bash
# Using docker-compose (if you're using Docker)
docker-compose ps

# Or check if Kafka is accessible
python -c "from kafka import KafkaAdminClient; KafkaAdminClient(bootstrap_servers=['localhost:29092']).close(); print('✓ Kafka is running')"
```

## Step 2: Install Dependencies

```bash
pip install -r requirements-kafka.txt
```

Or install manually:
```bash
pip install kafka-python pandas numpy scikit-learn joblib
```

## Step 3: Verify Setup

This will check that Kafka is running, topics exist, model is available, and data is accessible:

```bash
python test_pipeline.py --verify-only
```

Expected output:
```
✓ Kafka connection verified
✓ All required topics exist
✓ Model file found: telco-churn-production/src/models/sklearn_pipeline.pkl
✓ CSV file found: telco-churn-production/data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Step 4: Run the Full Pipeline Test

### Option A: Quick Test (Recommended for First Run)
```bash
python test_pipeline.py --mode streaming --duration 20
```

This will:
1. Start a consumer in the background
2. Start a producer that sends 10 events/sec for 20 seconds
3. Verify predictions are being published
4. Show summary statistics

### Option B: Batch Test
```bash
python test_pipeline.py --mode batch --batch-size 50 --num-batches 2
```

This will:
1. Send 100 messages in batches of 50
2. Consumer processes them in windows
3. Shows batch summary statistics

## Step 5: Run Producer and Consumer Separately

### Terminal 1 - Start Consumer First
```bash
# Streaming mode (runs indefinitely)
python consumer.py --mode streaming

# Or with duration limit
python consumer.py --mode streaming --duration 60
```

### Terminal 2 - Start Producer
```bash
# Streaming mode - 10 events/sec for 30 seconds
python producer.py --mode streaming --events-per-sec 10 --duration 30

# Or batch mode
python producer.py --mode batch --batch-size 100
```

## Step 6: Monitor the Pipeline

### View Predictions
```bash
kafka-console-consumer --bootstrap-server localhost:29092 --topic telco-churn-predictions --from-beginning
```

### View Raw Customer Data
```bash
kafka-console-consumer --bootstrap-server localhost:29092 --topic telco-raw-customers --from-beginning
```

### Check for Errors
```bash
kafka-console-consumer --bootstrap-server localhost:29092 --topic telco-deadletter --from-beginning
```

### Check Consumer Group Status
```bash
kafka-consumer-groups --bootstrap-server localhost:29092 --group telco-churn-consumer-group --describe
```

## Common Usage Patterns

### 1. Real-time Simulation
Simulate real-time customer data at 5 events/sec:
```bash
# Terminal 1
python consumer.py --mode streaming

# Terminal 2
python producer.py --mode streaming --events-per-sec 5
```

### 2. Batch Processing
Process entire dataset in batches:
```bash
# Terminal 1
python consumer.py --mode batch --window-size 200

# Terminal 2
python producer.py --mode batch --batch-size 1000
```

### 3. High-throughput Test
Test with high event rate:
```bash
# Terminal 1
python consumer.py --mode streaming

# Terminal 2
python producer.py --mode streaming --events-per-sec 100 --duration 60
```

### 4. Resume After Failure
If batch producer fails, resume from checkpoint:
```bash
python producer.py --mode batch --resume
```

## Troubleshooting

**"Connection refused" error:**
```bash
# Make sure Kafka is running
docker-compose up -d  # if using Docker
# Or start your Kafka service
```

**"Model not found" error:**
```bash
# Check if model exists
ls -la telco-churn-production/src/models/sklearn_pipeline.pkl

# If not, you may need to copy it
cp telco-churn-production/src/models/sklearn_pipeline.pkl models/churn_model.pkl
```

**Topics don't exist:**
```bash
# Run test_pipeline.py which creates topics automatically
python test_pipeline.py --verify-only

# Or create manually (see KAFKA_README.md)
```

**Consumer not receiving messages:**
```bash
# Reset consumer group offset
kafka-consumer-groups --bootstrap-server localhost:29092 \
  --group telco-churn-consumer-group \
  --reset-offsets --to-earliest --all-topics --execute
```

## Next Steps

1. Review the logs in `kafka_pipeline.log`
2. Check the KAFKA_README.md for detailed documentation
3. Experiment with different streaming rates and batch sizes
4. Monitor Kafka metrics and consumer lag
5. Customize `config.py` for your specific needs

## Example Session

Here's a complete example session:

```bash
# 1. Verify setup
python test_pipeline.py --verify-only

# 2. Run quick test
python test_pipeline.py --mode streaming --duration 30

# 3. Open new terminal and view predictions live
kafka-console-consumer --bootstrap-server localhost:29092 \
  --topic telco-churn-predictions

# 4. Check logs
tail -f kafka_pipeline.log

# 5. Run batch test
python test_pipeline.py --mode batch --batch-size 100 --num-batches 3
```

## Performance Metrics to Watch

- **Producer:** Events/sec, messages sent, failures
- **Consumer:** Messages processed, predictions/sec, errors
- **Kafka:** Consumer lag, partition balance
- **Model:** Prediction latency, churn probability distribution

Enjoy your Kafka-based real-time churn prediction pipeline!
