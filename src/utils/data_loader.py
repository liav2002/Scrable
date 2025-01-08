import yaml
import pandas as pd
from typing import Tuple

from src.utils.logger import Logger


def load_data(logger: Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load source data from the data directory.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        A tuple containing:
            train_df: DataFrame for training data.
            test_df: DataFrame for test data.
            games_df: DataFrame for games data.
            turns_df: DataFrame for turns data.
    """
    logger.log("Loading data paths from configuration...")
    with open("config/file_paths.yaml", "r") as file:
        paths = yaml.safe_load(file)["data_paths"]

    logger.log("Reading train.csv...")
    train_df = pd.read_csv(paths["train"])

    logger.log("Reading test.csv...")
    test_df = pd.read_csv(paths["test"])

    logger.log("Reading games.csv...")
    games_df = pd.read_csv(paths["games"])

    logger.log("Reading turns.csv...")
    turns_df = pd.read_csv(paths["turns"])

    logger.log("Data loading complete.")
    return train_df, test_df, games_df, turns_df


def load_best_model_path() -> str:
    with open("config/file_paths.yaml", "r") as file:
        return yaml.safe_load(file)["output_path"]["best_model"]