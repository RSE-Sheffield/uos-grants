from openai import OpenAI
import json
import os
import io
from datetime import datetime
from typing import Dict, List
import pandas as pd

import asyncio

from uos_grants.utils import (
    get_metadata_from_url_sql,
    get_content_string_from_url_sql,
    embed_researchers_to_pgvector,
    chunk_list,
    make_doc_from_sql,
)

from sqlalchemy import select

from uos_grants.connectors.db import get_session
from uos_grants.connectors.models import Researcher as ModelResearcher


class SingleBatchResult:
    def __init__(
        self,
        custom_id: str,
        embedding: List[float],
        metadata: Dict[str, str],
        content: str = None,
    ):
        self.custom_id = custom_id
        self.embedding = embedding
        self.metadata = metadata
        self.content = content

    def to_dict(self):
        return {
            "custom_id": self.custom_id,
            "embedding": self.embedding,
            "content": self.content,
            "metadata": self.metadata,
        }


class BatchOutput:
    def __init__(self):
        self.results = []

    def add_result(self, result: SingleBatchResult):
        self.results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        data = [result.to_dict() for result in self.results]
        return pd.DataFrame(data)


class BatchEmbeddingOpenAI:
    """
    Class to handle batch embedding using OpenAI API.
    ######################
    # TO BE PLACED IN AN EMBEDDING MODULE SOMEWHERE
    ######################
    """

    def __init__(self, open_ai_api_key: str = None, jsonl_file=None):
        self.client = OpenAI() if open_ai_api_key else OpenAI()
        self.input_content_strings: List[Dict[str, str]] = []
        self.batch_input_file = io.BytesIO()
        self.batch_results = BatchOutput()
        self.batch_job = None
        self.jsonl_file = jsonl_file

    def add_content_string(self, content: str, id: str):
        self.input_content_strings.append({"content": content, "id": id})

    def _create_payload(self, content: str, id: str):
        payload = {
            "custom_id": f"{id}",  # {str(metadata)}",
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
            id = data["id"]
            payload = self._create_payload(content, id)

            line = json.dumps(payload) + "\n"

            # Write to disk for logging/debugging
            if self.jsonl_file:
                self.jsonl_file.write(line)
            # Write to in-memory file
            self.batch_input_file.write(
                (json.dumps(payload) + "\n").encode("utf-8")
            )

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

    async def get_batch_job_results(self) -> pd.DataFrame:
        if self.get_batch_job_status() != "completed":
            raise ValueError("Batch job is not completed.")

        output_file_id = self.client.batches.retrieve(
            self.batch_job.id
        ).output_file_id
        output_file = self.client.files.content(output_file_id).text

        for line in output_file.split("\n"):
            if not line.strip():
                continue
            data = json.loads(line)
            custom_id = data["custom_id"]
            embedding = data["response"]["body"]["data"][0]["embedding"]
            try:
                metadata = await get_metadata_from_url_sql(custom_id)
                content = await get_content_string_from_url_sql(custom_id)
            except Exception as e:
                print(f"Error fetching metadata for {custom_id}: {e}")
                continue

            result = SingleBatchResult(custom_id, embedding, metadata, content)
            self.batch_results.add_result(result)

        return self.batch_results

    async def wait_for_batch_completion(self, check_interval: int = 2):
        current_status = None
        while True:
            status = self.get_batch_job_status()
            if status != current_status:
                print(
                    f"Batch job status {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: {status}"
                )
                current_status = status

            if status == "completed":
                print("Batch job completed! Fetching results...")
                return await self.get_batch_job_results()
            elif status == "failed":
                raise RuntimeError("Batch job failed.")
            elif status in ["cancelled", "expired"]:
                raise RuntimeError(f"Batch job {status}.")

            await asyncio.sleep(check_interval)


async def batch_and_embed_researchers(open_ai_api_key: str = None):
    """
    Process and embed researchers in batches.
    """
    async for session in get_session():
        researchers = await session.execute(select(ModelResearcher))
        researchers = researchers.scalars().all()
        researcher_docs = [
            make_doc_from_sql(researcher) for researcher in researchers
        ]
        # Create a batch embedding object
        batch_folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        content_dict = {}
        chunked_researchers = list(chunk_list(researcher_docs, 1000))
        for batch_idx, researcher_batch in enumerate(
            chunked_researchers, start=1
        ):
            try:
                os.makedirs(
                    f"/app/payload_batches/{batch_folder_name}", exist_ok=True
                )
                jsonl_file = open(
                    f"/app/payload_batches/{batch_folder_name}/batch_payload_{batch_idx}_of_{len(chunked_researchers)}.jsonl",
                    "a",
                )
            except Exception as e:
                print(
                    f"Error writing to file: {e} Trying to write to local folder."
                )
                os.makedirs(f"payload_batches/{batch_folder_name}", exist_ok=True)
                jsonl_file = open(
                    f"payload_batches/{batch_folder_name}/batch_payload_{batch_idx}_of_{len(chunked_researchers)}.jsonl",
                    "a",
                )
            try:
                print(
                    f"\n🔹 Processing Batch {batch_idx} of {len(chunked_researchers)} ({len(researcher_batch)} researchers)"
                )
                batch = BatchEmbeddingOpenAI(jsonl_file=jsonl_file)

                for researcher in researcher_batch:
                    batch.add_content_string(
                        content=researcher.page_content,
                        id=researcher.metadata["url"],
                    )
                    content_dict[researcher.metadata["url"]] = (
                        researcher.page_content
                    )

                batch.create_batch_job()
                # results_list = []
                results = await batch.wait_for_batch_completion()
                await embed_researchers_to_pgvector(results)
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                return batch
