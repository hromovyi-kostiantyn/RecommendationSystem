import os
import logging
import logging.config
import sys
from typing import Optional, Dict, Any

# Store configured loggers to avoid duplicate setup
_loggers = {}


def configure_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Configure structured logging with different levels for different components.

    Args:
        config: Configuration dictionary with logging settings

    Returns:
        Logger specifically for tracking results
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Define logging configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': 'logs/recommendation_system.log',
                'mode': 'w',
            },
            'results_file': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'filename': 'results/results.log',
                'mode': 'w',
            },
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': True
            },
            'data': {  # Data processing logs
                'handlers': ['file'],
                'level': 'INFO',
                'propagate': False
            },
            'models': {  # Model training logs
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': False
            },
            'evaluation': {  # Evaluation logs
                'handlers': ['console', 'file', 'results_file'],
                'level': 'INFO',
                'propagate': False
            },
            'results': {  # Results-only logger
                'handlers': ['console', 'results_file'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }

    # Ensure log directories exist
    os.makedirs('logs', exist_ok=True)

    # Apply configuration
    logging.config.dictConfig(logging_config)

    # Create a special logger just for results
    results_logger = logging.getLogger('results')

    return results_logger


def setup_logger(
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None,
        log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with appropriate handlers and formatters.

    Args:
        name: Name of the logger
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, no file logging)
        log_format: Format string for log messages

    Returns:
        Configured logger instance
    """
    # Check if logger already exists to avoid duplicate handlers
    if name in _loggers:
        return _loggers[name]

    # Create logger
    logger = logging.getLogger(name)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Convert string level to logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = level_map.get(level.upper(), logging.INFO)

    # Set logger level
    logger.setLevel(log_level)

    # Default format if none provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Store logger to avoid duplicate setup
    _loggers[name] = logger

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Name of the logger

    Returns:
        Logger instance
    """
    # Check if logger already exists
    if name in _loggers:
        return _loggers[name]

    # Create new logger
    return setup_logger(name)


def get_results_logger() -> logging.Logger:
    """
    Get the special results logger.

    Returns:
        Results logger instance
    """
    return logging.getLogger('results')