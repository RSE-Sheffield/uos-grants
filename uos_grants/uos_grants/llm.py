# llm.py

from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


class LLM:
    """
    LLM class to handle multiple LLM providers from config.

    Supported providers:

    - Ollama
    - OpenAI
    """

    def __init__(self, provider: str, model_name: str, api_key: str = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.llm = self._initialise_model()

    def _initialise_model(self):
        if self.provider == "ollama":
            return OllamaLLM(model=self.model_name)
        elif self.provider == "openai":
            return ChatOpenAI(model=self.model_name, api_key=self.api_key)
        elif self.provider == "google":
            return ChatGoogleGenerativeAI(
                model=self.model_name, api_key=self.api_key
            )
        else:
            raise ValueError(f"Provider {self.provider} is not supported")

    def __call__(self):
        return self.llm

    # def invoke(self, messages):
    #     return self.llm.invoke(messages)

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying LLM.

        Args:
            name (str): The attribute name.

        Returns:
            The attribute from the underlying LLM if it exists.
        """
        if hasattr(self.llm, name):
            return getattr(self.llm, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
