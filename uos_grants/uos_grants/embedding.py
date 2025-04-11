# embedding.py

import ast
import asyncio
import io
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI



class EmbeddingModel:
    """
    EmbeddingModel class to handle multiple embedding providers from config.

    Supported providers:

    - Ollama
    - OpenAI
    """

    def __init__(self, provider: str, model_name: str, api_key: str = None, dimensions: int = 3072):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.dimensions = dimensions 
        self.embedding_model = self._initialise_model()

    def _initialise_model(self):
        if self.provider == "ollama":
            return OllamaEmbeddings(model=self.model_name)
        elif self.provider == "openai":
            return OpenAIEmbeddings(
                model=self.model_name, api_key=self.api_key
            )
        else:
            raise ValueError(f"Provider {self.provider} is not supported.")

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying embedding object.

        Args:
            name (str): The attribute name.

        Returns:
            The attribute from the underlying embedding if it exists.
        """
        if hasattr(self.embedding_model, name):
            return getattr(self.embedding_model, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


class BatchEmbeddingOpenAIOld:
    def __init__(self, open_ai_api_key: str = None):
        if open_ai_api_key:
            self.client = OpenAI(api_key=open_ai_api_key)
        else:
            self.client = OpenAI()
        self.input_content_paths: Dict[str, List[str]] = []
        self.input_content_strings: Dict[str, str | Dict[str, str]] = []
        self.batch_input_file = io.BytesIO()
        self.batch_results = None

    def add_content_file(self, file_path: str = None):
        """
        Add a file path to the list of input content paths for embedding.

        Args:
            file_path (str): The path to the file containing content for embedding.
            metadata (Optional[Dict[str, str]], optional): Metadata associated with the file. Defaults to None.
        """
        self.input_content_paths.append(file_path)

    def add_content_string(
        self, content: str, metadata: Optional[Dict[str, str]] = None
    ):
        """
        Add a string containing content for embedding, along with optional metadata.

        Args:
            content (str): The content to be embedded as a string.
            metadata (Optional[Dict[str, str]], optional): Metadata associated with the content. Defaults to None.
        """
        self.input_content_strings.append(content, metadata)

    def _create_payload(
        self,
        content: str,
        metadata: Optional[dict] = None,
        id: Optional[str] = str(uuid.uuid4()),
        dimensions: int = 1024,
    ):
        """
        Create a payload for the OpenAI API.

        Args:
            content (str): The content to be embedded as a string.
            id (Optional[str], optional): A custom id for the embedding. Defaults to str(uuid.uuid4()).
            dimensions (int, optional): Embedding dimensions (Dependant on the embedding model). Defaults to 1024 for text-embedding-003-large.
        """

        payload = {
            "custom_id": id + "|" + str(metadata) if metadata else id,
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": "text-embedding-3-large",
                "input": content,
                "encoding_format": "float",
                "dimensions": dimensions,
            },
        }

        return payload

    def _create_jsonl_file(self):
        """Create a JSONL file from the input content paths."""
        if not self.input_content_paths and not self.input_content_strings:
            raise ValueError("No content to create JSONL file.")
        if self.input_content_paths:
            for file_path in self.input_content_paths:
                with open(file_path, "r") as file:
                    data = json.load(file)
                    content = data["documents"]
                    metadata = data["metadatas"]
                    payload = self._create_payload(
                        content=content,
                        id=f"{os.path.basename(file_path)}",
                    )
                    self.batch_input_file.write(
                        (json.dumps(payload) + "\n").encode("utf-8")
                    )
        if self.input_content_strings:
            for data in self.input_content_strings:
                content = data["content"]
                metadata = data["metadata"]
                if metadata["id"] is None:
                    try:
                        metadata["id"] = (
                            metadata["url"]
                            .replace("https://www.sheffield.ac.uk/", "")
                            .replace("/", "_")
                        )
                    except KeyError:
                        metadata["id"] = str(uuid.uuid4())
                payload = self._create_payload(
                    content=content,
                    metadata=metadata,
                    id=data["metadata"]["id"],
                )
                self.batch_input_file.write(
                    (json.dumps(payload) + "\n").encode("utf-8")
                )

    def _add_file_to_batch(self):
        """Add a file to the batch for embedding."""
        self.batch_input = self.client.files.create(
            file=self.batch_input_file,
            purpose="batch",
        )

    def create_batch_job(self):
        """Create a batch job for the given file ID."""
        self._create_jsonl_file()
        self._add_file_to_batch()
        self.batch_job = self.client.batches.create(
            input_file_id=self.batch_input.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )

    def get_batch_job_status(self):
        """Get the status of the batch job."""
        return self.client.batches.retrieve(self.batch_job.id).status

    def get_batch_job_results(self) -> pd.DataFrame:
        """Get the results of the batch job."""
        if not isinstance(self.batch_results, pd.DataFrame):
            if self.get_batch_job_status() != "completed":
                raise ValueError(
                    "Batch job is not completed. Please check the status."
                )
            if self.get_batch_job_status() == "in_progress":
                raise ValueError(
                    "Batch job is still in progress. Please check the status."
                )
            if self.get_batch_job_status() == "failed":
                raise ValueError("Batch job failed. Please check the status.")

            self.output_file_id = self.client.batches.retrieve(
                self.batch_job.id
            ).output_file_id
            self.output_file = self.client.files.content(
                self.output_file_id
            ).text

            embedding_results = []
            for line in self.output_file.split("\n")[:-1]:
                data = json.loads(line)
                custom_id = data["custom_id"]
                if "|" in custom_id:
                    custom_id, metadata = custom_id.split("|", 1)
                    metadata = ast.literal_eval(metadata)
                else:
                    metadata = {}
                embedding = data["response"]["body"]["data"][0]["embedding"]
                embedding_results.append([custom_id, embedding, metadata])

            self.batch_results = pd.DataFrame(
                embedding_results,
                columns=["custom_id", "embedding", "metadata"],
            )
            return self.batch_results
        elif not self.batch_job:
            raise ValueError(
                "No batch job created. Please create a batch job first."
            )
        else:
            return self.batch_results

    async def wait_for_batch_completion(self, check_interval: int = 30):
        """
        Asynchronously checks the batch job status until it completes.
        Once completed, it automatically fetches the results.

        Args:
            check_interval (int): Time in seconds between status checks (default: 30).
        """
        while True:
            status = self.get_batch_job_status()
            print(
                f"Batch job status {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: {status}"
            )

            if status == "completed":
                print("Batch job completed! Fetching results...")
                results = self.get_batch_job_results()
                print("Results retrieved successfully.")
                return results  # Returns a DataFrame

            elif status == "failed":
                raise RuntimeError(
                    "Batch job failed. Please check logs for details."
                )

            elif status in ["cancelled", "expired"]:
                raise RuntimeError(
                    f"Batch job status: {status}. Stopping checks."
                )

            # Non-blocking sleep before checking again
            await asyncio.sleep(check_interval)

    def get_batch_job_logs(self):
        """Get the logs of the batch job."""
        if not self.batch_job:
            raise ValueError(
                "No batch job created. Please create a batch job first."
            )
        return self.client.batches.retrieve(self.batch_job.id)


class BatchEmbeddingOpenAI:
    def __init__(self, open_ai_api_key: str = None):
        self.client = OpenAI(api_key=open_ai_api_key) if open_ai_api_key else OpenAI()
        self.input_content_strings: List[Dict[str, str]] = []
        self.batch_input_file = io.BytesIO()
        self.batch_results = None
        self.batch_job = None

    def add_content_string(self, content: str, metadata: Dict[str, str]):
        self.input_content_strings.append({"content": content, "metadata": metadata})

    def _create_payload(self, content: str, metadata: dict, id: str):
        payload = {
            "custom_id": f"{id}|{str(metadata)}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": "text-embedding-3-large",
                "input": content,
                "encoding_format": "float",
                "dimensions": 3072,
            },
        }
        return payload

    def _create_jsonl_file(self):
        if not self.input_content_strings:
            raise ValueError("No content to create JSONL file.")

        for data in self.input_content_strings:
            content = data["content"]
            metadata = data["metadata"]
            id = metadata.get("id") or metadata.get("url", str(uuid.uuid4())).replace("https://www.sheffield.ac.uk/", "").replace("/", "_")
            payload = self._create_payload(content, metadata, id)

            line = json.dumps(payload) + "\n"

            # Write to disk for logging/debugging
            with open(f"batch_payload_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl", "a") as f:
                f.write(line)
            # Write to in-memory file
            self.batch_input_file.write((json.dumps(payload) + "\n").encode("utf-8"))

    def _add_file_to_batch(self):
        self.batch_input_file.seek(0)
        self.batch_input = self.client.files.create(
            file=self.batch_input_file,
            purpose="batch",
        )

    def create_batch_job(self):
        self._create_jsonl_file()
        self._add_file_to_batch()
        self.batch_job = self.client.batches.create(
            input_file_id=self.batch_input.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )

    def get_batch_job_status(self):
        return self.client.batches.retrieve(self.batch_job.id).status

    def get_batch_job_results(self) -> pd.DataFrame:
        if not isinstance(self.batch_results, pd.DataFrame):
            if self.get_batch_job_status() != "completed":
                raise ValueError("Batch job is not completed.")

            output_file_id = self.client.batches.retrieve(self.batch_job.id).output_file_id
            output_file = self.client.files.content(output_file_id).text

            embedding_results = []
            for line in output_file.split("\n"):
                if not line.strip():
                    continue
                data = json.loads(line)
                print(data.keys())
                custom_id = data["custom_id"]
                if "|" in custom_id:
                    custom_id, metadata = custom_id.split("|", 1)
                    metadata = ast.literal_eval(metadata)
                else:
                    metadata = {}
                embedding = data["response"]["body"]["data"][0]["embedding"]
                embedding_results.append([custom_id, embedding, metadata])

            self.batch_results = pd.DataFrame(embedding_results, columns=["custom_id", "embedding", "metadata"])
        return self.batch_results

    async def wait_for_batch_completion(self, check_interval: int = 30):
        current_status = None
        while True:
            status = self.get_batch_job_status()
            if status != current_status:
                print(f"Batch job status {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: {status}")
                current_status = status

            if status == "completed":
                print("Batch job completed! Fetching results...")
                return self.get_batch_job_results()
            elif status == "failed":
                raise RuntimeError("Batch job failed.")
            elif status in ["cancelled", "expired"]:
                raise RuntimeError(f"Batch job {status}.")

            await asyncio.sleep(check_interval)
