import logging
import sys
import os

def get_logger(name: str) -> logging.Logger:
    """Creates and returns a configured logger."""
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if logger already exists
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler('logs/project.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger