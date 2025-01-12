from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

LOG_DIR = "logs"
CONFIG_PATH = "config/config.yaml"
CONFIG_FILE_PATHS = "config/file_paths.yaml"
FALSE_ANALYSIS_DIR = "output/false_analysis_df/"
FALSE_ANALYSIS_FILE = "false_analysis.csv"
MODEL_2_CLASS_MAP = {"XGBoost": XGBRegressor, "Linear Regression": LinearRegression,
                             "Neural Network": MLPRegressor}