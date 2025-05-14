import os
import logging
import sys
from typing import Optional, Dict, Any

# Logging levels mapping
LOGGING_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Store configured loggers to avoid duplicate setup
_loggers = {}


def setup_logger(
        name: str,
        config: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        log_file: Optional[str] = None,
        log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure a logger.

    Args:
        name: Name of the logger
        config: Configuration dictionary with logging settings
        level: Log level (overrides config if provided)
        log_file: Path to log file (overrides config if provided)
        log_format: Log format string (overrides config if provided)

    Returns:
        Configured logger instance
    """
    # Check if logger already exists to avoid duplicate handlers
    if name in _loggers:
        return _loggers[name]

    # Create logger
    logger = logging.getLogger(name)

    # Clear any existing handlers (in case logger exists but isn't in _loggers)
    if logger.handlers:
        logger.handlers.clear()

    # Get configurations
    if config is None:
        config = {}

    # Determine log level
    log_level_str = level or config.get('level', 'INFO')
    numeric_level = LOGGING_LEVELS.get(log_level_str.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Determine log format
    format_str = log_format or config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date_format = config.get('date_format', '%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter(format_str, datefmt=date_format)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    file_path = log_file or config.get('file')
    if file_path:
        # Ensure directory exists
        log_dir = os.path.dirname(file_path)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError as e:
                # Log to console since file logging isn't set up yet
                print(f"Error creating log directory {log_dir}: {e}", file=sys.stderr)

        try:
            file_handler = logging.FileHandler(file_path, mode='a')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Log to console since file handler failed
            print(f"Error setting up file handler for {file_path}: {e}", file=sys.stderr)
            # Continue without file logging

    # Store logger to avoid duplicate setup
    _loggers[name] = logger

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one.

    This function checks if a logger with the given name already exists.
    If it does, returns the existing logger. Otherwise, creates a new one
    with default settings.

    Args:
        name: Name of the logger (typically __name__ from the calling module)

    Returns:
        Logger instance
    """
    # Check if logger exists
    if name in _loggers:
        return _loggers[name]

    # Try to get logging configuration from config files
    config = _get_logging_config()

    # Create and return logger
    return setup_logger(name, config)


def _get_logging_config() -> Dict[str, Any]:
    """
    Get logging configuration from config files.

    Attempts to load configuration from config files. If that fails,
    returns default configuration.

    Returns:
        Dictionary with logging configuration
    """
    # Default configuration
    default_config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'date_format': '%Y-%m-%d %H:%M:%S',
    }

    # Try to load configuration from files
    try:
        # In a real application, this would load from config files
        # For now, just use the default configuration
        pass
    except Exception:
        # If loading fails, use default configuration
        pass

    return default_config


# Create root logger
root_logger = get_logger("recommendation_system")