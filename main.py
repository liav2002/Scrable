import os
import yaml
from datetime import datetime
from utils.data_loader import load_data
from utils.config_loader import load_config
from utils.logger import Logger, FileLogger
from utils.model_manager import run_models_and_get_best, get_model_instance, train_and_predict_with_best
from model.pipeline.pipeline import DataPipeline

logger = Logger()


def main():
    """
    Entry point for the application.
    """
    runtime_signature = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = f"logs/{runtime_signature}.log"
    file_logger = FileLogger(log_file_path)
    logger.subscribe(file_logger)

    logger.log("Application started.")
    config = load_config()
    logger.log("Configuration loaded.\n")

    train_df, test_df, games_df, turns_df = load_data(logger)
    logger.log("Data loaded successfully.\n")

    pipeline = DataPipeline(config["bots_and_scores"], turns_df, games_df)
    logger.log("pipeline initialized.")
    processed_train_df = pipeline.process_train_data(train_df)
    processed_test_df = pipeline.process_test_data(test_df)
    logger.log("Data processing completed.\n")

    # Define models to evaluate
    models = {
        model_name: get_model_instance(
            model_details["class"],
            model_details["params"]
        )
        for model_name, model_details in config["models"].items()
    }
    logger.log("models initialized.")

    # Run models and find the best model
    results_df, best_model = run_models_and_get_best(processed_train_df, models, config, logger)
    logger.log("model evaluation completed.\n")
    logger.log("model Performance Summary:")
    logger.log(results_df.to_string(index=False) + "\n")

    # Train and predict with the best model
    test_predictions = train_and_predict_with_best(processed_train_df, processed_test_df, best_model, logger)
    logger.log("First 10 predictions:")
    for idx, pred in enumerate(test_predictions[:10]):
        logger.log(f"Prediction {idx + 1}: {pred:.4f}")

    logger.log("Application completed.")


if __name__ == "__main__":
    main()
