import os
import pickle
from datetime import datetime

from utils.data_loader import load_data
from utils.config_loader import load_config
from utils.logger import Logger, FileLogger
from utils.model_manager import run_models_and_get_best, get_model_instance

from model.pipeline.pipeline import DataPipeline


class Solver:
    def __init__(self):
        """
        Initialize the Solver class with necessary configurations and logger setup.
        """
        self.logger = Logger()
        self.config = None
        self.train_df = None
        self.test_df = None
        self.games_df = None
        self.turns_df = None
        self.best_model_path = "./output/best_model.pkl"
        self._setup()

    def _setup(self):
        """
        Private method to set up logging, configuration, and data loading.
        """
        runtime_signature = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = f"logs/{runtime_signature}.log"
        file_logger = FileLogger(log_file_path)
        self.logger.subscribe(file_logger)

        self.logger.log("Application started.")
        self.config = load_config()
        self.logger.log("Configuration loaded.")

        self.train_df, self.test_df, self.games_df, self.turns_df = load_data(self.logger)
        self.logger.log("Data loaded successfully.")

    def find_best_model(self):
        """
        Find the best model by evaluating all models, train the best model on the full dataset,
        and save it as a pickle file.
        """
        self.logger.log("Initializing data pipeline for training.")
        pipeline = DataPipeline(self.config["bots_and_scores"], self.turns_df, self.games_df)
        processed_train_df = pipeline.process_train_data(self.train_df)
        self.logger.log("Data processing completed.")

        # Define models to evaluate
        models = {}
        if self.config["hyperparameter_tuning"]["search_best_params"]:
            self.logger.log("Performing hyperparameter tuning for all models.")
            for model_name, model_details in self.config["models"].items():
                model = get_model_instance(
                    model_details["class"],
                    model_details["params"]
                )
                if hasattr(model, "search_best_params"):
                    self.logger.log(f"Starting hyperparameter tuning for {model_name}.")
                    best_params = model.search_best_params(
                        processed_train_df.drop(columns=["user_rating"]),
                        processed_train_df["user_rating"],
                        self.config
                    )
                    self.logger.log(f"Best parameters for {model_name}: {best_params}")
                    model_details["params"] = best_params
                models[model_name] = model
        else:
            self.logger.log("Using predefined model parameters from configuration.")
            models = {
                model_name: get_model_instance(
                    model_details["class"],
                    model_details["params"]
                )
                for model_name, model_details in self.config["models"].items()
            }

        self.logger.log("Models initialized.")

        # Run models and find the best model
        results_df, best_model = run_models_and_get_best(processed_train_df, models, self.config, self.logger)
        self.logger.log("Model evaluation completed.")
        self.logger.log("Model Performance Summary:")
        self.logger.log(results_df.to_string(index=False))

        # Train the best model on the entire training data
        self.logger.log("Training the best model on the full dataset.")
        best_model.fit(
            processed_train_df.drop(columns=["user_rating"]),
            processed_train_df["user_rating"]
        )

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)

        # Save the best model as a pickle file
        with open(self.best_model_path, "wb") as f:
            pickle.dump(best_model, f)
        self.logger.log(f"Best model saved to {self.best_model_path}.")

    def test_best_model(self):
        """
        Load the best model, process the test data, and make predictions.
        """
        if not os.path.exists(self.best_model_path):
            self.logger.log(
                f"Error: Best model file '{self.best_model_path}' not found. Please run 'find_best_model' first.")
            return

        # Load the best model from pickle
        self.logger.log(f"Loading the best model from {self.best_model_path}.")
        with open(self.best_model_path, "rb") as f:
            best_model = pickle.load(f)

        # Process the test data
        self.logger.log("Initializing data pipeline for testing.")
        pipeline = DataPipeline(self.config["bots_and_scores"], self.turns_df, self.games_df)
        processed_test_df = pipeline.process_test_data(self.test_df)
        self.logger.log("Test data processing completed.")

        # Make predictions
        self.logger.log("Making predictions on the test data.")
        predictions = best_model.predict(processed_test_df.drop(columns=["user_rating"], errors="ignore"))

        # Print first 10 predictions
        self.logger.log("First 10 predictions:")
        for idx, pred in enumerate(predictions[:10]):
            self.logger.log(f"Prediction {idx + 1}: {pred:.4f}")
