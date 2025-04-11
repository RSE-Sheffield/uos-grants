import os
import tempfile

import pytest
from pydantic import ValidationError

from uos_grants.config.config_handler import AppConfig

# Configurations
VALID_OPENAI_CONFIG = """
embedding_model:
  name: "text-embedding-3-large"
  provider: "openai"
  api_key_env: "OPENAI_DUMMY_API_KEY"

llm:
  name: "gpt-4o-mini"
  provider: "openai"
  api_key_env: "OPENAI_DUMMY_API_KEY"
"""

VALID_OLLAMA_CONFIG = """
embedding_model:
  name: "llama3.1"
  provider: "ollama"

llm:
  name: "llama3.1"
  provider: "ollama"
"""

MISSING_FIELDS_CONFIG = """
embedding_model:
  name: "text-embedding-3-large"

llm:
  name: "gpt-4o-mini"
"""

MISSING_API_KEY_CONFIG = """
embedding_model:
  name: "text-embedding-3-large"
  provider: "openai"

llm:
  name: "gpt-4o-mini"
  provider: "openai"
"""


@pytest.fixture
def set_dummy_env():
    """Set dummy environment variables for testing."""
    os.environ["OPENAI_DUMMY_API_KEY"] = "dummy_key"
    yield
    del os.environ["OPENAI_DUMMY_API_KEY"]


def test_valid_openai_config(set_dummy_env):
    """Test a valid OpenAI configuration."""
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False
    ) as temp_file:
        temp_file.write(VALID_OPENAI_CONFIG.encode("utf-8"))
        temp_file_path = temp_file.name

    try:
        config = AppConfig.load(temp_file_path)
        assert config.embedding_model.provider == "openai"
        assert config.embedding_model.api_key == "dummy_key"
        assert config.llm.provider == "openai"
        assert config.llm.api_key == "dummy_key"
    finally:
        os.unlink(temp_file_path)


def test_valid_ollama_config():
    """Test a valid Ollama configuration."""
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False
    ) as temp_file:
        temp_file.write(VALID_OLLAMA_CONFIG.encode("utf-8"))
        temp_file_path = temp_file.name

    try:
        config = AppConfig.load(temp_file_path)
        assert config.embedding_model.provider == "ollama"
        assert config.embedding_model.api_key is None
        assert config.llm.provider == "ollama"
        assert config.llm.api_key is None
    finally:
        os.unlink(temp_file_path)


def test_missing_fields():
    """Test a configuration missing required fields."""
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False
    ) as temp_file:
        temp_file.write(MISSING_FIELDS_CONFIG.encode("utf-8"))
        temp_file_path = temp_file.name
    try:
        with pytest.raises(ValidationError):
            AppConfig.load(temp_file_path)
    finally:
        os.unlink(temp_file_path)


def test_missing_api_key():
    """Test a configuration missing 'api_key_env'."""
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False
    ) as temp_file:
        temp_file.write(MISSING_API_KEY_CONFIG.encode("utf-8"))
        temp_file_path = temp_file.name

    try:
        config = AppConfig.load(temp_file_path)
        assert config.embedding_model.api_key is None
        assert config.llm.api_key is None
    finally:
        os.unlink(temp_file_path)
