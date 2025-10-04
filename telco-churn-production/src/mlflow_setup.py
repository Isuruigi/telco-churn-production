import os
import subprocess
import time
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

def configure_tracking_uri(tracking_uri: str):
    """Sets the MLFLOW_TRACKING_URI environment variable."""
    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    print(f"Environment variable MLFLOW_TRACKING_URI set to: {tracking_uri}")

def setup_artifact_store(artifact_location: str):
    """
    Informs the user how to set up the artifact store.
    The artifact store is configured when the MLflow server is started.
    """
    print("--- Artifact Store Information ---")
    print(f"The artifact store should be configured when starting the MLflow server.")
    print(f"Use the --default-artifact-root option, for example:")
    print(f"mlflow server --default-artifact-root {artifact_location}")
    print("------------------------------------")

def initialize_mlflow_server(backend_store_uri: str, default_artifact_root: str, host: str = '127.0.0.1', port: int = 5000) -> subprocess.Popen:
    """
    Starts the MLflow tracking server as a background process.

    Args:
        backend_store_uri (str): The URI for the backend store (e.g., 'sqlite:///mlflow.db').
        default_artifact_root (str): The root directory for storing artifacts.
        host (str): The host to bind the server to.
        port (int): The port to run the server on.

    Returns:
        subprocess.Popen: The process object for the running server.
    """
    command = [
        'mlflow', 'server',
        '--backend-store-uri', backend_store_uri,
        '--default-artifact-root', default_artifact_root,
        '--host', host,
        '--port', str(port)
    ]
    
    print(f"Starting MLflow server with command: {' '.join(command)}")
    # Start the server as a background process
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5) # Give the server a moment to start

    if process.poll() is None:
        print(f"MLflow server started successfully on http://{host}:{port}")
    else:
        print("Error: MLflow server failed to start.")
        stdout, stderr = process.communicate()
        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())
        return None
    
    return process

def create_experiments(tracking_uri: str, experiment_names: list):
    """Creates one or more MLflow experiments if they do not already exist."""
    print("--- Creating/Verifying MLflow Experiments ---")
    client = MlflowClient(tracking_uri=tracking_uri)
    for name in experiment_names:
        try:
            experiment = client.get_experiment_by_name(name)
            if experiment is None:
                client.create_experiment(name)
                print(f"Experiment '{name}' created.")
            else:
                print(f"Experiment '{name}' already exists.")
        except MlflowException as e:
            print(f"Error with experiment '{name}': {e}")

def setup_model_registry(tracking_uri: str):
    """Informs the user about the model registry setup."""
    print("\n--- Model Registry Information ---")
    print("The MLflow Model Registry is automatically available when you run an MLflow tracking server.")
    print(f"You can access it through the UI and client APIs at: {tracking_uri}")
    print("------------------------------------")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="MLflow Setup Script for Production Environment")
    parser.add_argument('--backend_store', type=str, default="sqlite:///mlflow.db",
                        help="Backend store URI for MLflow (e.g., a database connection string).")
    parser.add_argument('--artifact_root', type=str, default="./mlruns",
                        help="Root directory for storing MLflow artifacts.")
    parser.add_argument('--host', type=str, default="127.0.0.1", help="Server host.")
    parser.add_argument('--port', type=int, default=5000, help="Server port.")
    parser.add_argument('--experiments', nargs='+', default=['prod-telco-churn-prediction', 'dev-churn-experiments'],
                        help="A list of experiment names to create.")

    args = parser.parse_args()

    # 1. Configure Tracking URI (can also be set as an environment variable)
    configure_tracking_uri(f"http://{args.host}:{args.port}")

    # 2. Start the MLflow Server
    # In a real production setup, you would run this as a persistent service (e.g., with systemd or in a container).
    mlflow_server_process = initialize_mlflow_server(
        backend_store_uri=args.backend_store,
        default_artifact_root=args.artifact_root,
        host=args.host,
        port=args.port
    )

    if mlflow_server_process:
        # 3. Create Experiments
        create_experiments(f"http://{args.host}:{args.port}", args.experiments)

        # 4. Display Model Registry Info
        setup_model_registry(f"http://{args.host}:{args.port}")

        print("\nMLflow setup is complete. The server is running in the background.")
        print("To stop the server, you will need to manually terminate the process with PID:", mlflow_server_process.pid)

        # Keep the script alive to keep the server running for a bit in this demo
        try:
            time.sleep(60) # Keep server alive for 60 seconds
        finally:
            print("\nStopping MLflow server...")
            mlflow_server_process.terminate()
