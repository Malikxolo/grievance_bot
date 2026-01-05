"""
Brain-Heart Deep Research System - Core Module
Simple version that definitely works with Python 3.12.3
"""

__version__ = "2.0.0"

# Define exceptions inline since exceptions.py is being removed
class BrainHeartException(Exception):
    """Base exception for Brain-Heart system"""
    pass

class LLMClientError(BrainHeartException):
    """LLM client related errors"""
    pass

class BrainAgentError(BrainHeartException):
    """Brain agent specific errors"""
    pass

class HeartAgentError(BrainHeartException):
    """Heart agent specific errors"""
    pass

class ToolExecutionError(BrainHeartException):
    """Tool execution errors"""
    pass

class ConfigurationError(BrainHeartException):
    """Configuration related errors"""
    pass

class APIKeyError(ConfigurationError):
    """API key related errors"""
    pass

class ModelNotAvailableError(LLMClientError):
    """Model not available error"""
    pass

# Import config - may fail if dependencies missing
try:
    from .config import Config
except ImportError:
    Config = None

# Import other components - may fail if dependencies missing
try:
    from .llm_client import LLMClient
except ImportError:
    LLMClient = None

try:
    from .tools import ToolManager
except ImportError:
    ToolManager = None

try:
    from .brain_agent import BrainAgent
except ImportError:
    BrainAgent = None

try:
    from .heart_agent import HeartAgent
except ImportError:
    HeartAgent = None

# Export everything that was imported successfully
__all__ = [
    'BrainHeartException',
    'LLMClientError', 
    'BrainAgentError',
    'HeartAgentError',
    'ToolExecutionError',
    'ConfigurationError',
    'APIKeyError',
    'ModelNotAvailableError'
]

# Add optional components if they loaded
if Config is not None:
    __all__.append('Config')
if LLMClient is not None:
    __all__.append('LLMClient')
if ToolManager is not None:
    __all__.append('ToolManager')
if BrainAgent is not None:
    __all__.append('BrainAgent')
if HeartAgent is not None:
    __all__.append('HeartAgent')