import yaml
from utils.data_loader import load_data
from utils.model_runner import run_models
from utils.config_loader import load_config
from Model.Pipeline.pipeline import DataPipeline
from Model.Models.regression_model import RegressionModel


def main():
    """
    Entry point for the application.
    """
    # Load configuration
    config = load_config()

    # Load data
    train_df, test_df, games_df, turns_df = load_data()

    # Initialize and run pipeline
    pipeline = DataPipeline(config["bots_and_scores"], turns_df, games_df)
    processed_train_df = pipeline.process_train_data(train_df)
    processed_test_df = pipeline.process_test_data(test_df)

    # Run and evaluate models
    results_df, best_model_name, best_rmse, test_predictions = run_models(
        processed_train_df, processed_test_df, RegressionModel()
    )

    # Display results
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))
    print(f"\nBest Model: {best_model_name} (RMSE: {best_rmse:.4f})")
    print("\nSample Test Predictions:")
    for idx, pred in enumerate(test_predictions[:10]):
        print(f"Test Sample {idx + 1}: Predicted Rating = {pred:.2f}")


if __name__ == "__main__":
    main()
