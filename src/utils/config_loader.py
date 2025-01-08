import yaml

def load_config(file_path: str = "Config/config.yaml") -> dict:
    """
    Load the configuration file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration data as a dictionary.
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)
