"""
Core package - Configuration, settings, and shared utilities.
"""

from app.core.config import settings
from app.core.device import get_device, get_device_info

__all__ = ["settings", "get_device", "get_device_info"]
