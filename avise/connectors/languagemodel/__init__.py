from .base import BaseLMConnector, Message
from .ollama import OllamaLMConnector
from .openai import OpenAILMConnector
from .generic import GenericRESTLMConnector
from .lehmus_connector import LehmusLMConnector

# Category name shown by `avise --connector-list`
DISPLAY_NAME = "Language Model"
