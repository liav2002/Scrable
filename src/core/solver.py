import os
import pickle
from tabulate import tabulate
from datetime import datetime

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from src.utils.config_loader import load_config
from src.utils.logger import logger, FileLogger
from src.utils.data_loader import load_data, load_best_model_path

from src.model.model_handler import ModelHandler
from src.model.pipeline.pipeline import DataPipeline
from src.model.model_manager import run_models_and_get_best

from config.consts import LOG_DIR, CONFIG_PATH, FALSE_ANALYSIS_DIR


class Solver:
    """
    The Solver class is responsible for orchestrating the end-to-end machine learning pipeline for the Scrabble project.
    It handles configuration setup, data loading, model initialization, hyperparameter tuning, training, evaluation,
    and prediction on test data. The class also manages logging to ensure traceability throughout the process.
    """

    def __init__(self):
        """
        Initialize the Solver class with necessary configurations and logger setup.
        """
        self.config = self.train_df = self.test_df = self.games_df = self.turns_df = self.best_model_path = None
        self._setup()

    def _setup(self):
        """
        Private method to set up logging, configuration, and data loading.
        """
        runtime_signature = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = LOG_DIR + f"/{runtime_signature}.log"
        file_logger = FileLogger(log_file_path)
        logger.subscribe(file_logger)

        logger.log("Application started.")
        self.config = load_config(CONFIG_PATH)
        logger.log("Configuration loaded.")

        self.train_df, self.test_df, self.games_df, self.turns_df = load_data()
        logger.log("Data loaded successfully.")

        self.best_model_path = load_best_model_path()

    def find_best_model(self):
        """
        Find the best model by evaluating all models, train the best model on the full dataset,
        and save it as a pickle file.
        """
        logger.log("Initializing data pipeline for training.")
        pipeline = DataPipeline(self.config["bots_and_scores"], self.turns_df, self.games_df)
        processed_train_df = pipeline.process_train_data(self.train_df)
        logger.log("Data processing completed.")

        # Initialize model handlers
        models = {}
        for model_name, model_details in self.config["models"].items():
            if model_name == "XGBoost":
                model_type = XGBRegressor
            elif model_name == "Linear Regression":
                model_type = LinearRegression
            elif model_name == "Neural Network":
                model_type = MLPRegressor
            else:
                raise ValueError(f"Unsupported model type: {model_name}")

            model_handler = ModelHandler(model=model_type, params=model_details["params"])

            if self.config["hyperparameter_tuning"]["search_best_params"]:
                logger.log(f"Performing hyperparameter tuning for {model_name}.")
                best_params = model_handler.search_best_params(
                    processed_train_df.drop(columns=["user_rating"]),
                    processed_train_df["user_rating"],
                    self.config["hyperparameter_tuning"]["search_space"],
                    self.config["cross_validation"]["cv_folds"],
                    self.config["cross_validation"]["scoring"]["rmse"],
                    self.config["hyperparameter_tuning"]["trials"]
                )
                logger.log(f"Best parameters for {model_name}: {best_params}")
                model_handler.params.update(best_params)

            models[model_name] = model_handler

        logger.log("Models initialized.")

        # Run models and find the best model
        results_df, best_model = run_models_and_get_best(processed_train_df, models, self.config)
        logger.log("Model evaluation completed.")
        logger.log("Model Performance Summary:")
        logger.log(tabulate(results_df, headers="keys", tablefmt="pretty"))

        # Train the best model on the entire training data
        logger.log("Training the best model on the full dataset.")
        best_model.fit(
            processed_train_df.drop(columns=["user_rating"]),
            processed_train_df["user_rating"]
        )

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)

        # Save the best model as a pickle file
        with open(self.best_model_path, "wb") as f:
            pickle.dump(best_model.model, f)
        logger.log(f"Best model saved to {self.best_model_path}.")

    def test_best_model(self):
        """
        Load the best model, process the test data, and make predictions.
        """
        if not os.path.exists(self.best_model_path):
            logger.log(
                f"Error: Best model file '{self.best_model_path}' not found. Please run 'find_best_model' first.")
            raise FileNotFoundError(
                f"Best model file '{self.best_model_path}' not found. Please run 'find_best_model' first."
            )

        # Load the best model from pickle
        logger.log(f"Loading the best model from {self.best_model_path}.")
        with open(self.best_model_path, "rb") as f:
            best_model = pickle.load(f)

        # Process the test data
        logger.log("Initializing data pipeline for testing.")
        pipeline = DataPipeline(self.config["bots_and_scores"], self.turns_df, self.games_df)
        processed_test_df = pipeline.process_test_data(self.test_df)
        logger.log("Test data processing completed.")

        # Make predictions
        logger.log("Making predictions on the test data.")
        predictions = best_model.predict(processed_test_df.drop(columns=["user_rating"], errors="ignore"))

        # Print first 10 predictions
        logger.log("First 10 predictions:")
        for idx, pred in enumerate(predictions[:10]):
            logger.log(f"Prediction {idx + 1}: {pred:.4f}")

    def false_analysis(self):
        """
        Perform false analysis on the model's predictions, identifying the most significant errors.
        """
        if not os.path.exists(self.best_model_path):
            logger.log(
                f"Error: Best model file '{self.best_model_path}' not found. Please run 'find_best_model' first.")
            raise FileNotFoundError(
                f"Best model file '{self.best_model_path}' not found. Please run 'find_best_model' first."
            )

        # Load the best model
        logger.log(f"Loading the best model from {self.best_model_path}.")
        with open(self.best_model_path, "rb") as f:
            best_model = pickle.load(f)

        # Initialize the pipeline and process the training data
        logger.log("Initializing data pipeline for false analysis.")
        pipeline = DataPipeline(self.config["bots_and_scores"], self.turns_df, self.games_df)
        processed_train_df = pipeline.process_train_data(self.train_df)
        logger.log("Data pipeline processing for training data completed.")

        # Make predictions on the processed training data
        logger.log("Making predictions on the training data for false analysis.")
        predictions = best_model.predict(processed_train_df.drop(columns=["user_rating"]))

        # Add new columns to the dataframe
        logger.log("Adding 'predicted_user_rating' and 'error' columns.")
        processed_train_df["predicted_user_rating"] = predictions
        processed_train_df["error"] = (
                processed_train_df["user_rating"] - processed_train_df["predicted_user_rating"]
        ).abs()

        # Sort by the error column in descending order
        logger.log("Sorting the dataframe by 'error' in descending order.")
        sorted_df = processed_train_df.sort_values(by="error", ascending=False)

        # Save the dataframe to the output directory
        logger.log(f"Saving sorted dataframe to {FALSE_ANALYSIS_DIR}.")
        os.makedirs(FALSE_ANALYSIS_DIR, exist_ok=True)
        output_path = os.path.join(FALSE_ANALYSIS_DIR, "false_analysis.csv")
        sorted_df.to_csv(output_path, index=False)
        logger.log(f"False analysis dataframe saved to {output_path}.")
