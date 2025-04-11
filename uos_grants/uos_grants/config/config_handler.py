import os

import yaml
from pydantic import BaseModel


class EmbeddingModelConfig(BaseModel):
    provider: str
    name: str
    api_key_env: str | None = None
    api_key: str | None = None

    @property
    def api_key(self) -> str:
        """Retrieve the API key from the environment."""
        if self.api_key:
            return self.api_key
        if self.api_key_env is None:
            return None
        api_key = os.getenv(self.api_key_env)
        if not self.api_key:
            return None
        return api_key


class LLMConfig(BaseModel):
    provider: str
    name: str
    api_key_env: str | None = None
    api_key: str | None = None

    @property
    def api_key(self) -> str:
        """Retrieve the API key from the environment."""
        if self.api_key:
            return self.api_key
        if self.api_key_env is None:
            return None
        api_key = os.getenv(self.api_key_env)
        if not self.api_key:
            return None
        return api_key

class AppConfig(BaseModel):
    embedding_model: EmbeddingModelConfig
    llm: LLMConfig

    @staticmethod
    def load(config_path: str = None) -> "AppConfig":
        """
        Load configuration from multiple sources:
        1. User-provided config file (default: ./config.yaml)
        2. Default config file (packaged with the library).
        """
        # Default paths
        user_config_path = config_path or os.path.expanduser("./config.yaml")
        # default_config_path = os.path.join(
        #     os.path.dirname(__file__), "default_config.yaml"
        # )

        # Load user config if it exists
        if os.path.exists(user_config_path):
            with open(user_config_path, "r") as file:
                config_data = yaml.safe_load(file)
        else:
            raise FileNotFoundError(
                f"Config file not found at {user_config_path}"
            )
            # # Fall back to the default config
            # with open(default_config_path, "r") as file:
            #     config_data = yaml.safe_load(file)

        return AppConfig(**config_data)
