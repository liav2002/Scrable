import os
import yaml
from datetime import datetime
from utils.data_loader import load_data
from utils.config_loader import load_config
from utils.logger import Logger, FileLogger
from utils.model_runner import run_models, get_model_instance
from Model.Pipeline.pipeline import DataPipeline

logger = Logger()


def main():
    """
    Entry point for the application.
    """
    # Generate runtime signature for the log file
    runtime_signature = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = f"logs/{runtime_signature}.log"

    # Attach a file logger
    file_logger = FileLogger(log_file_path)
    logger.subscribe(file_logger)

    logger.log("Application started.")
    config = load_config()
    logger.log("Configuration loaded.")

    train_df, test_df, games_df, turns_df = load_data()
    logger.log("Data loaded successfully.")

    pipeline = DataPipeline(config["bots_and_scores"], turns_df, games_df)
    logger.log("Pipeline initialized.")
    processed_train_df = pipeline.process_train_data(train_df)
    processed_test_df = pipeline.process_test_data(test_df)
    logger.log("Data processing completed.")

    models = {
        model_name: get_model_instance(model_details["class"])
        for model_name, model_details in config["models"].items()
    }
    logger.log("Models initialized.")

    results_df, best_model_name, best_rmse, test_predictions = run_models(
        processed_train_df, processed_test_df, models, config
    )
    logger.log("Model evaluation completed.")

    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))
    logger.log(f"Best Model: {best_model_name} with RMSE: {best_rmse:.4f}")

    logger.log("Application completed.")


if __name__ == "__main__":
    main()
