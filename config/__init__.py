import os
import yaml
import logging
from typing import Dict, Any, Optional


class Config:
    """Configuration manager that loads and provides access to configuration settings."""

    def __init__(self, config_path: str):
        """
        Initialize the Config object.

        Args:
            config_path: Path to the configuration file.
        """
        self.config_path = config_path
        self.config_data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Dict containing configuration settings.
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                logging.info(f"Configuration loaded from {self.config_path}")
                return config
        except Exception as e:
            logging.error(f"Error loading configuration from {self.config_path}: {str(e)}")
            raise

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key.
            default: Default value if key doesn't exist.

        Returns:
            Configuration value.
        """
        keys = key.split('.')
        value = self.config_data

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logging.warning(f"Configuration key '{key}' not found, using default: {default}")
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key.
            value: Value to set.
        """
        keys = key.split('.')
        config = self.config_data

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        logging.debug(f"Configuration key '{key}' set to '{value}'")

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the current configuration to a file.

        Args:
            path: Path to save the configuration file. If None, use the original path.
        """
        save_path = path or self.config_path

        try:
            with open(save_path, 'w') as file:
                yaml.safe_dump(self.config_data, file, default_flow_style=False)
            logging.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logging.error(f"Error saving configuration to {save_path}: {str(e)}")
            raise

    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.

        Args:
            base_config: Base configuration.
            override_config: Configuration to override base.

        Returns:
            Merged configuration.
        """
        merged = base_config.copy()

        for key, value in override_config.items():
            if (key in base_config and isinstance(base_config[key], dict)
                    and isinstance(value, dict)):
                merged[key] = Config.merge_configs(base_config[key], value)
            else:
                merged[key] = value

        return merged


def load_config(config_name: str = "config_base.yaml") -> Config:
    """
    Load a configuration file by name.

    Args:
        config_name: Name of the configuration file.

    Returns:
        Config object.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, "config")

    # Check if config is in experiments directory
    if config_name != "config_base.yaml" and not os.path.exists(os.path.join(config_dir, config_name)):
        config_path = os.path.join(config_dir, "config_experiments", config_name)
    else:
        config_path = os.path.join(config_dir, config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = Config(config_path)

    # If not base config, merge with base config
    if config_name != "config_base.yaml":
        base_config_path = os.path.join(config_dir, "config_base.yaml")
        base_config = Config(base_config_path)

        # Merge configs
        merged_data = Config.merge_configs(
            base_config.config_data,
            config.config_data
        )
        config.config_data = merged_data

    return config