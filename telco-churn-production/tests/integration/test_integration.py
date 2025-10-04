import pytest
import pandas as pd
import numpy as np
import os

# Add src directory to path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_pipeline import DataPipeline
from src.training_pipeline import TrainingPipeline
from src.inference_pipeline import InferencePipeline
from src.pipeline_orchestrator import PipelineOrchestrator
from config.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

@pytest.fixture(scope="module")
def mock_raw_data_file(tmpdir_factory):
    """Creates a temporary raw CSV file for integration testing."""
    data = {
        'customerID': [f'{i}' for i in range(100)],
        'gender': np.random.choice(['Female', 'Male'], 100),
        'tenure': np.random.randint(1, 72, 100),
        'MonthlyCharges': np.random.uniform(20, 120, 100),
        'TotalCharges': [str(x) for x in np.random.uniform(20, 8000, 100)],
        'Churn': np.random.choice(['No', 'Yes'], 100, p=[0.8, 0.2])
    }
    for col in CATEGORICAL_FEATURES:
        if col not in data:
            data[col] = ['No'] * 100
    for col in NUMERICAL_FEATURES:
        if col not in data:
            data[col] = [0] * 100

    df = pd.DataFrame(data)
    file_path = tmpdir_factory.mktemp("data").join("raw_data.csv")
    df.to_csv(file_path, index=False)
    return str(file_path)

class TestDataPipeline:
    def test_run(self, mock_raw_data_file):
        data_pipe = DataPipeline(data_path=mock_raw_data_file)
        result = data_pipe.run()
        assert len(result) == 5 # X_train, X_test, y_train, y_test, preprocessor
        assert result[0] is not None # X_train
        assert result[1] is not None # X_test

class TestTrainingAndInferencePipelines:
    @pytest.fixture
    def trained_pipeline_path(self, tmp_path, mock_raw_data_file):
        """Runs the training pipeline to produce a model for inference tests."""
        # Override the default MODELS_PATH to use a temporary directory
        from config import config
        original_models_path = config.MODELS_PATH
        config.MODELS_PATH = str(tmp_path)

        data_pipe = DataPipeline(data_path=mock_raw_data_file)
        training_pipe = TrainingPipeline(data_pipe)
        # In a real test, you might want to mock the tuning step to speed it up
        training_pipe.run()
        
        model_path = os.path.join(str(tmp_path), 'final_pipeline.joblib')
        
        # Restore original path
        config.MODELS_PATH = original_models_path
        
        return model_path

    def test_training_pipeline_creates_model(self, trained_pipeline_path):
        assert os.path.exists(trained_pipeline_path)

    def test_inference_pipeline(self, trained_pipeline_path, mock_raw_data_file):
        inference_pipe = InferencePipeline(model_path=trained_pipeline_path)
        
        # Get a sample from the mock data for prediction
        sample_df = pd.read_csv(mock_raw_data_file, nrows=1)
        sample_dict = sample_df.to_dict(orient='records')[0]

        prediction, confidence = inference_pipe.predict_single(sample_dict)
        assert prediction in [0, 1]
        assert 0.5 <= confidence <= 1.0

class TestPipelineOrchestrator:
    def test_health_check(self, mock_raw_data_file):
        # This test will fail if a model doesn't exist, which is expected on a clean run
        orchestrator = PipelineOrchestrator(data_path=mock_raw_data_file)
        # We expect it to fail on the model check but pass on the data check
        assert not orchestrator.validate_pipeline_health()

    # A full run test can be slow and is often done in dedicated integration test environments
    # @pytest.mark.slow
    # def test_full_run(self, mock_raw_data_file):
    #     orchestrator = PipelineOrchestrator(data_path=mock_raw_data_file)
    #     orchestrator.run_complete_pipeline()
    #     assert os.path.exists(orchestrator.final_model_path)
