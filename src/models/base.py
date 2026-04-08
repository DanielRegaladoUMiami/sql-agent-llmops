"""
Base model class defining the interface for all specialized models.

All model implementations inherit from BaseModel and implement
the abstract methods for loading and generating outputs.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all model implementations."""

    def __init__(self, model_name: str, model_path: Optional[str] = None) -> None:
        """
        Initialize base model.

        Args:
            model_name: Name/identifier of the model
            model_path: Path to model weights or config
        """
        self.model_name = model_name
        self.model_path = model_path
        self.is_loaded = False
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self) -> None:
        """
        Load the model and initialize it for inference.

        Must be implemented by subclasses. Should set self.model
        and update self.is_loaded flag.

        Raises:
            Exception: If model loading fails
        """
        pass

    @abstractmethod
    def generate(self, **kwargs) -> Any:
        """
        Generate output from the model.

        Method signature varies by model type. Subclasses must implement.

        Returns:
            Model-specific output (string, dict, etc.)
        """
        pass

    def unload(self) -> None:
        """Unload model to free memory."""
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        logger.info(f"Model {self.model_name} unloaded")

    def _validate_loaded(self) -> None:
        """Validate that model is loaded before inference."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"Model {self.model_name} is not loaded. Call load() first.")

    def __repr__(self) -> str:
        """String representation of model."""
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.__class__.__name__}(name={self.model_name}, status={status})"
