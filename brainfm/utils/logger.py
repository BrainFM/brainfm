import logging
import os
from datetime import datetime

def get_logger(log_dir: str = "./logs", experiment_name=None) -> logging.Logger:
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate a unique log filename using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    if experiment_name:
        log_filename = os.path.join(log_dir, f"log_{experiment_name}_{timestamp}.log")
    else:
        log_filename = os.path.join(log_dir, f"log_{timestamp}.log")

    # Create a logger
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)  # Log all levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Set up formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")

    # File handler to write logs to a file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)  # Save all logs to the file
    file_handler.setFormatter(formatter)

    # Console handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Print INFO and above to the console
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger