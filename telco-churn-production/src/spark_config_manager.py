from pyspark.sql import SparkSession

class SparkConfigManager:
    """Provides optimized Spark configuration for ML workloads."""

    def __init__(self, app_name="TelcoChurnML", master="local[*]"):
        """
        Initializes the SparkConfigManager.

        Args:
            app_name (str): The name of the Spark application.
            master (str): The Spark master URL.
        """
        self.app_name = app_name
        self.master = master
        self.spark_builder = SparkSession.builder.appName(self.app_name).master(self.master)
        self.spark_session = None

    def configure_spark_for_ml(self):
        """Applies common configurations optimized for ML workloads."""
        self.spark_builder.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        self.spark_builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
        print("Applied ML-specific Spark configurations.")
        return self

    def set_spark_memory_config(self, driver_memory="4g", executor_memory="4g"):
        """Sets the memory configuration for the Spark driver and executors."""
        self.spark_builder.config("spark.driver.memory", driver_memory)
        self.spark_builder.config("spark.executor.memory", executor_memory)
        print(f"Set driver memory to {driver_memory} and executor memory to {executor_memory}.")
        return self

    def set_executor_config(self, executor_cores=2, executor_instances=2):
        """Sets the configuration for the Spark executors."""
        self.spark_builder.config("spark.executor.cores", str(executor_cores))
        self.spark_builder.config("spark.executor.instances", str(executor_instances))
        print(f"Set executor cores to {executor_cores} and instances to {executor_instances}.")
        return self

    def initialize_spark_session(self) -> SparkSession:
        """Builds and initializes the Spark session with the specified configurations."""
        if self.spark_session is None:
            print("--- Initializing Spark Session ---")
            self.spark_session = self.spark_builder.getOrCreate()
            print("Spark session initialized successfully.")
            print(f"Spark version: {self.spark_session.version}")
        else:
            print("Spark session already initialized.")
        return self.spark_session

    def stop_spark_session(self):
        """Stops the active Spark session."""
        if self.spark_session:
            print("--- Stopping Spark Session ---")
            self.spark_session.stop()
            self.spark_session = None
            print("Spark session stopped.")

if __name__ == '__main__':
    # --- Demonstration of SparkConfigManager ---
    
    # 1. Create and configure a Spark session
    spark_manager = SparkConfigManager(app_name="SparkDemo", master="local[2]")
    
    spark_manager.set_spark_memory_config(driver_memory="2g", executor_memory="2g") \
                 .set_executor_config(executor_cores=1, executor_instances=2) \
                 .configure_spark_for_ml()

    # 2. Initialize the session
    spark = spark_manager.initialize_spark_session()

    # 3. Use the Spark session (example)
    if spark:
        data = [("Alice", 34), ("Bob", 45), ("Catherine", 29)]
        columns = ["name", "age"]
        df = spark.createDataFrame(data, columns)
        print("\n--- Sample Spark DataFrame ---")
        df.show()
        print(f"Number of partitions: {df.rdd.getNumPartitions()}")

    # 4. Stop the session
    spark_manager.stop_spark_session()
