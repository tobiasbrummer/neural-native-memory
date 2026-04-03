"""
I/O utilities for KV-Embedding experiments.

Handles JSON persistence, logging setup, and result organization.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def setup_logging(
    experiment_name: str,
    results_dir: Path,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Configure logging for an experiment.
    
    Args:
        experiment_name: Name for the logger
        results_dir: Directory to store log file
        console_level: Console output level
        file_level: File output level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / f"{experiment_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. File: {log_file}")
    
    return logger


def create_results_dir(experiment_prefix: str, base_dir: Path = None) -> Path:
    """
    Create a timestamped results directory.
    
    Args:
        experiment_prefix: Prefix for directory name (e.g., 'exp1')
        base_dir: Base directory (default: data/results/)
    
    Returns:
        Path to created directory
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "data" / "results"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = base_dir / f"{experiment_prefix}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return results_dir


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save data to JSON file with numpy support.
    
    Args:
        data: Dictionary to save
        filepath: Target file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Source file path
    
    Returns:
        Loaded dictionary
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_embeddings(
    embeddings: np.ndarray,
    filepath: Path,
    compress: bool = True,
) -> None:
    """
    Save embeddings to numpy file.
    
    Args:
        embeddings: Embedding array
        filepath: Target file path
        compress: Whether to use compression
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        np.savez_compressed(filepath, embeddings=embeddings)
    else:
        np.save(filepath, embeddings)


def load_embeddings(filepath: Path) -> np.ndarray:
    """
    Load embeddings from numpy file.
    
    Args:
        filepath: Source file path
    
    Returns:
        Embedding array
    """
    filepath = Path(filepath)
    
    if filepath.suffix == ".npz":
        data = np.load(filepath)
        return data["embeddings"]
    else:
        return np.load(filepath)


def load_test_documents(filepath: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load synthetic test documents.
    
    Args:
        filepath: Path to test documents JSON (default: data/synthetic/test_docs.json)
    
    Returns:
        List of document dictionaries
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "synthetic" / "test_docs.json"
    
    return load_json(filepath)
