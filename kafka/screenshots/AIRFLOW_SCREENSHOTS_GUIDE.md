# Airflow DAG Screenshots Guide - Steps 10-12

Complete guide for capturing Airflow UI screenshots for the Kafka integration project.

---

## 📋 Prerequisites

Before capturing screenshots:

1. **Activate your environment:**
   ```powershell
   conda activate telco-churn
   ```

2. **Ensure Kafka is running:**
   ```powershell
   docker ps | findstr kafka
   ```
   Should show kafka and zookeeper containers running.

3. **Deploy Kafka DAGs to Airflow:**
   ```powershell
   # Set AIRFLOW_HOME (if not already set)
   $env:AIRFLOW_HOME = "$HOME\airflow"

   # Create dags directory if it doesn't exist
   mkdir -Force $env:AIRFLOW_HOME\dags

   # Copy DAG files
   Copy-Item "kafka\airflow\dags\kafka_streaming_dag.py" -Destination "$env:AIRFLOW_HOME\dags\"
   Copy-Item "kafka\airflow\dags\kafka_batch_dag.py" -Destination "$env:AIRFLOW_HOME\dags\"
   ```

4. **Verify DAGs are recognized:**
   ```powershell
   airflow dags list | Select-String "kafka"
   ```
   Should show:
   - kafka_batch_pipeline
   - kafka_streaming_pipeline

---

## 🚀 Start Airflow

You need TWO PowerShell terminals:

### **Terminal 1: Start Webserver**
```powershell
conda activate telco-churn
cd "C:\Users\IPK\Telco churn project 1"
airflow webserver --port 8080
```

Wait for message: `Running the Gunicorn Server with...`

### **Terminal 2: Start Scheduler**
```powershell
conda activate telco-churn
cd "C:\Users\IPK\Telco churn project 1"
airflow scheduler
```

Wait for message: `Starting the scheduler`

### **Access Airflow UI**

1. Open browser: **http://localhost:8080**
2. Login with:
   - **Username:** `admin`
   - **Password:** (the password you created during `airflow users create`)

---

## 📸 Screenshot 10: Airflow DAGs List Page

**What to capture:** Main DAGs list page showing both Kafka DAGs

### Steps:

1. **Navigate to DAGs page:**
   - Open browser: http://localhost:8080
   - You should see the main DAGs list

2. **Enable both Kafka DAGs:**
   - Find `kafka_streaming_pipeline` in the list
   - Toggle the switch on the left side to **ON** (blue)
   - Find `kafka_batch_pipeline` in the list
   - Toggle the switch to **ON** (blue)

3. **Search for your DAGs (optional but recommended):**
   - In the search box at top, type: `kafka`
   - This will filter to show only your 2 Kafka DAGs

4. **Prepare the view:**
   - Make sure both DAGs are visible on screen
   - You should see columns: DAG name, Owner, Runs, Schedule, Last Run, etc.
   - Both DAGs should show toggle switches as ON (blue)

5. **Capture screenshot:**
   - Press `Windows Key + Shift + S` to open Snipping Tool
   - Select and capture the entire DAGs list area
   - Save as: `kafka\screenshots\airflow_logs\10_airflow_dags_list.png`

**What the screenshot should show:**
- ✅ Both `kafka_streaming_pipeline` and `kafka_batch_pipeline` visible
- ✅ Toggle switches showing ON (blue/enabled)
- ✅ DAG schedule information (streaming: None, batch: 0 * * * *)
- ✅ Tags: kafka, batch, ml, churn-prediction, streaming

---

## 📸 Screenshot 11: Streaming DAG Graph View

**What to capture:** Graph visualization of kafka_streaming_pipeline

### Steps:

1. **Navigate to Streaming DAG:**
   - From the DAGs list page, click on `kafka_streaming_pipeline`
   - This opens the DAG detail view

2. **Switch to Graph View:**
   - Look at the top tabs: Grid, Graph, Calendar, etc.
   - Click on **"Graph"** tab
   - Wait for the graph to render

3. **Understand the graph layout:**
   You should see these tasks connected:
   ```
   [check_kafka] ──┐
   [verify_topics] ─┼──> [start_consumer] ──> [health_check] ──> [monitor_metrics]
   [verify_model] ──┘                             │
                                                  │
                                                  └──> [cleanup_on_failure]
   ```

4. **Optional: Trigger a test run (to show task colors):**
   - Click the **Play button** (▶) in top right corner
   - Click "Trigger DAG" to confirm
   - Wait a few seconds for tasks to start running
   - Tasks will change colors:
     - **Green** = Success
     - **Light Green** = Running
     - **Grey** = Queued/Pending
     - **Red** = Failed

5. **Capture screenshot:**
   - Make sure the entire graph is visible
   - Zoom in/out if needed using browser zoom (Ctrl + Mouse Wheel)
   - Press `Windows Key + Shift + S`
   - Capture the graph area showing all tasks and connections
   - Save as: `kafka\screenshots\airflow_logs\11_streaming_dag_graph.png`

**What the screenshot should show:**
- ✅ All 7 tasks visible: check_kafka, verify_topics, verify_model, start_consumer, health_check, monitor_metrics, cleanup_on_failure
- ✅ Task dependencies (arrows showing flow)
- ✅ Task colors (if you triggered a run)
- ✅ DAG name "kafka_streaming_pipeline" at top

---

## 📸 Screenshot 12: Batch DAG Graph View

**What to capture:** Graph visualization of kafka_batch_pipeline

### Steps:

1. **Navigate to Batch DAG:**
   - Click the **"DAGs"** link at top to go back to DAGs list
   - Click on `kafka_batch_pipeline`
   - This opens the Batch DAG detail view

2. **Switch to Graph View:**
   - Click on **"Graph"** tab at the top
   - Wait for the graph to render

3. **Understand the graph layout:**
   You should see this flow:
   ```
   [check_prerequisites]
           │
           ▼
   [run_producer]
           │
           ▼
   [run_consumer]
           │
           ▼
   [parse_summary]
           │
           ▼
   [generate_report]
           │
           ▼
   [check_threshold] ──┬──> [send_success_notification] ──┐
                        │                                   │
                        └──> [send_failure_alert] ─────────┤
                                                            ▼
                                                       [cleanup]
   ```

4. **Optional: Trigger a test run:**
   - Click the **Play button** (▶) in top right
   - Optionally modify parameters or keep defaults:
     - batch_size: 100
     - window_size: 100
     - num_windows: 5
   - Click "Trigger DAG"
   - Wait for tasks to execute (this will take a few minutes)
   - Tasks will turn green as they complete

5. **Capture screenshot:**
   - Ensure all tasks are visible in the graph
   - Zoom to fit if needed (Ctrl + Mouse Wheel)
   - Press `Windows Key + Shift + S`
   - Capture the complete graph showing all 8 tasks
   - Save as: `kafka\screenshots\airflow_logs\12_batch_dag_graph.png`

**What the screenshot should show:**
- ✅ All 8 tasks visible: check_prerequisites, run_producer, run_consumer, parse_summary, generate_report, check_threshold, send_success_notification/send_failure_alert, cleanup
- ✅ Task dependencies and branching logic (check_threshold branches to success or failure)
- ✅ Task colors if you ran the DAG
- ✅ DAG name "kafka_batch_pipeline" at top

---

## 🎯 Summary Checklist

After completing all 3 screenshots, verify you have:

- [ ] **Screenshot 10:** `kafka\screenshots\airflow_logs\10_airflow_dags_list.png`
  - Shows both Kafka DAGs in the main list
  - Both DAGs are enabled (toggle ON)

- [ ] **Screenshot 11:** `kafka\screenshots\airflow_logs\11_streaming_dag_graph.png`
  - Shows kafka_streaming_pipeline graph
  - All 7 tasks visible with connections

- [ ] **Screenshot 12:** `kafka\screenshots\airflow_logs\12_batch_dag_graph.png`
  - Shows kafka_batch_pipeline graph
  - All 8 tasks visible with branching logic

---

## 🛠️ Troubleshooting

### Issue: DAGs not appearing in Airflow UI

**Solution 1: Check DAG files were copied**
```powershell
dir $env:AIRFLOW_HOME\dags\kafka*.py
```

**Solution 2: Check for Python errors**
```powershell
python "$env:AIRFLOW_HOME\dags\kafka_batch_dag.py"
python "$env:AIRFLOW_HOME\dags\kafka_streaming_dag.py"
```
Should run without errors.

**Solution 3: Check scheduler logs**
```powershell
# Look for errors in scheduler terminal
# Or check logs:
type "$env:AIRFLOW_HOME\logs\scheduler\latest\*.log"
```

**Solution 4: Restart Airflow**
- Stop both terminals (Ctrl+C)
- Start webserver and scheduler again

---

### Issue: Path errors in DAG logs

**Problem:** DAGs can't find kafka scripts (producer.py, consumer.py)

**Solution:** Update path in DAG files

Edit both DAG files and ensure KAFKA_DIR is correct:

**For Windows:**
```python
# In kafka_streaming_dag.py and kafka_batch_dag.py
# Replace the path calculation with absolute path:
KAFKA_DIR = r'C:\Users\IPK\Telco churn project 1\kafka'
PROJECT_ROOT = r'C:\Users\IPK\Telco churn project 1'
sys.path.insert(0, KAFKA_DIR)
```

Then re-copy the updated files:
```powershell
Copy-Item "kafka\airflow\dags\*.py" -Destination "$env:AIRFLOW_HOME\dags\" -Force
```

---

### Issue: Can't login to Airflow UI

**Solution: Create admin user**
```powershell
airflow users create `
    --username admin `
    --firstname Admin `
    --lastname User `
    --role Admin `
    --email admin@example.com `
    --password admin
```

Then login with username: `admin`, password: `admin`

---

### Issue: Module import errors in task logs

**Solution: Install dependencies in correct environment**
```powershell
conda activate telco-churn
pip install apache-airflow==2.7.0 kafka-python psutil
```

---

## 📊 Bonus: Additional Screenshots (Optional)

If you want to capture more for documentation:

### **Task Execution Logs:**
1. Trigger the batch DAG
2. Click on a running/completed task (e.g., `run_consumer`)
3. Click **"Log"** button
4. Capture the log output
5. Save as: `kafka\screenshots\airflow_logs\task_execution_logs.png`

### **DAG Run Grid View:**
1. Click on batch DAG
2. Stay on **"Grid"** tab (default view)
3. Shows timeline of DAG runs with task status
4. Capture and save as: `kafka\screenshots\airflow_logs\dag_run_grid_view.png`

### **Gantt Chart:**
1. Click on batch DAG
2. Click **"Gantt"** tab
3. Shows task duration and overlap
4. Capture and save as: `kafka\screenshots\airflow_logs\gantt_chart.png`

---

## 📞 Reference: Project 1 Screenshots

Your Project 1 screenshots for comparison:
- `telco-churn-production\Screenshots\Airflow DAGs.png` - Similar to Screenshot 10
- `telco-churn-production\Screenshots\DAG Graph Visualization.png` - Similar to Screenshots 11 & 12
- `telco-churn-production\Screenshots\Task Excecution Logs.png` - Bonus screenshot
- `telco-churn-production\Screenshots\Trigger DAGs.png` - Bonus screenshot

Follow the same style and capture quality!

---

## ✅ Quick Command Reference

```powershell
# 1. Activate environment
conda activate telco-churn

# 2. Deploy DAGs
$env:AIRFLOW_HOME = "$HOME\airflow"
mkdir -Force $env:AIRFLOW_HOME\dags
Copy-Item "kafka\airflow\dags\*.py" -Destination "$env:AIRFLOW_HOME\dags\"

# 3. Verify DAGs
airflow dags list | Select-String "kafka"

# 4. Start Airflow (2 terminals)
# Terminal 1:
airflow webserver --port 8080

# Terminal 2:
airflow scheduler

# 5. Access UI
# Browser: http://localhost:8080
# Username: admin
# Password: (your password)
```

---

**Good luck with your screenshots! 📸✨**

*Last Updated: October 16, 2025*
