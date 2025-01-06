import yaml
from utils.data_loader import load_data
from utils.config_loader import load_config
from utils.model_runner import run_models, get_model_instance
from Model.Pipeline.pipeline import DataPipeline


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

    # Define models to evaluate
    models = {
        model_name: get_model_instance(model_details["class"])
        for model_name, model_details in config["models"].items()
    }

    # Run models and evaluate
    results_df, best_model_name, best_rmse, test_predictions = run_models(
        processed_train_df, processed_test_df, models, config
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
