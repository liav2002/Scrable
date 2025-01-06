import yaml
import pandas as pd
from typing import Tuple

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    with open("Config/file_paths.yaml", "r") as file:
        paths = yaml.safe_load(file)["data_paths"]

    return (
        pd.read_csv(paths["train"]),
        pd.read_csv(paths["test"]),
        pd.read_csv(paths["games"]),
        pd.read_csv(paths["turns"]),
    )
