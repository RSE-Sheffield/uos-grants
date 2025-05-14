from itertools import islice

from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_core.messages import trim_messages

import os
from contextlib import aclosing

from .connectors.models import (
    Researcher as ModelResearcher,
    StaffUrls as ModelStaffUrls,
)

from .connectors.db import get_session

import requests
from sqlalchemy import select

from .connectors.repository import (
    update_researcher,
    insert_researcher,
    update_staff_url,
)
from .researchers import Researcher

from .webloaders import logger
from typing import List, Dict, Optional

import tiktoken


def chunk_list(iterable, chunk_size=500):
    """Yield successive n-sized chunks from an iterable."""
    iterable = iter(iterable)
    while chunk := list(islice(iterable, chunk_size)):
        yield chunk


async def embed_researchers_to_pgvector(results):  #: BatchOutput):
    collection_name = os.getenv(
        "VECTOR_DB_COLLECTION", "sheffield_researchers"
    )
    vector_db = PGVector(
        collection_name=collection_name,
        connection=os.getenv(
            "VECTOR_DB_URI",
            "postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
        ),
        use_jsonb=True,
        embeddings=None,
    )
    results_df = results.to_dataframe()
    ids = results_df["custom_id"].tolist()
    embeddings = results_df["embedding"].tolist()
    metadatas = results_df["metadata"].tolist()
    content = results_df["content"].tolist()
    vector_db.add_embeddings(
        texts=content,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


async def add_url_to_staff_db(url: str):
    """Add a URL to the staff database."""
    async with aclosing(get_session()) as db_gen:
        async for db in db_gen:
            async with db.begin():
                result = await db.execute(
                    ModelStaffUrls.__table__.select().where(
                        ModelStaffUrls.url == url.strip()
                    )
                )
                existing = result.scalar_one_or_none()

                if existing:
                    # logger.info(f"[UPDATE] {url}")
                    continue
                else:
                    db.add(
                        ModelStaffUrls(url=url.strip(), last_response="200")
                    )


async def add_staff_to_researcher_db():
    """Add staff URLs to the researcher database."""
    async for session in get_session():
        result = await session.execute(select(ModelStaffUrls.url))
        all_urls = result.scalars().all()

        for url in all_urls:
            try:
                response = requests.get(url)
            except requests.exceptions.TooManyRedirects:
                continue
            # logger.info(f"Processing {url}")
            if response.status_code == 200:
                html = response.text
                researcher = Researcher(html=html, url=url)
                try:
                    data = researcher.to_dict()
                    data["last_response"] = str(response.status_code)

                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    continue
                existing_researcher = await session.execute(
                    select(ModelResearcher).where(ModelResearcher.url == url)
                )
                existing_researcher = existing_researcher.scalar_one_or_none()
                await update_staff_url(session, url, str(response.status_code))
                if existing_researcher:
                    logger.info(f"[UPDATE] {url}")
                    await update_researcher(session, existing_researcher, data)
                else:
                    logger.info(f"[INSERT] {url}")
                    await insert_researcher(session, data)
            else:
                # Handle the case where the URL is not reachable
                existing_researcher = await session.execute(
                    select(ModelResearcher).where(ModelResearcher.url == url)
                )
                existing_researcher = existing_researcher.scalar_one_or_none()
                await update_staff_url(session, url, str(response.status_code))
                data = {"last_response": str(response.status_code)}
                if existing_researcher:
                    logger.info(f"[UPDATE] {url}")
                    await update_researcher(session, existing_researcher, data)
                else:
                    logger.info(f"[INSERT] {url}")
                    await insert_researcher(session, data)
                await update_staff_url(session, url, str(response.status_code))
                logger.error(f"Failed to fetch {url}: {response.status_code}")

        await session.commit()


def make_researcher_str(researcher: ModelResearcher) -> str:
    """
    Create a Researcher object from a SQL.
    """
    researcher = researcher.__dict__
    fields = [
        f"Name: {researcher['name']}" if researcher["name"] else "",
        (
            f"Department: {researcher['department']}"
            if researcher["department"]
            else ""
        ),
        f"URL: {researcher['url']}" if researcher["url"] else "",
        (
            f"Main Role: {researcher['main_role']}"
            if researcher["main_role"]
            else ""
        ),
        (
            f"Additional Roles: {researcher['additional_roles']}"
            if researcher["additional_roles"]
            else ""
        ),
        f"Email: {researcher['email']}" if researcher["email"] else "",
        (
            f"Telephone: {researcher['telephone']}"
            if researcher["telephone"]
            else ""
        ),
        f"Profile: {researcher['profile']}" if researcher["profile"] else "",
    ]
    researcherstr = "\n".join(filter(None, fields))
    return researcherstr


def make_doc_from_sql(researcher: ModelResearcher) -> Document:
    """
    Create a Researcher object from a SQL.
    """
    researcherstr = make_researcher_str(researcher)
    metadata = {
        k: v
        for k, v in researcher.__dict__.items()
        if k
        not in [
            "profile",
            "research_interests",
            "additional_roles",
            "_sa_instance_state",
            "last_response",
            "id",
        ]
        and v is not None
    }
    return Document(
        page_content=researcherstr,
        metadata=metadata,
    )


async def get_metadata_from_url_sql(url: str) -> Optional[ModelResearcher]:
    """
    Get metadata from a URL in the SQL database.
    """
    async for session in get_session():
        result = await session.execute(
            select(ModelResearcher).where(ModelResearcher.url == url)
        )
        metadata = result.scalars().first()
        metadata_dict = metadata.__dict__
        metadata_dict.pop("_sa_instance_state", None)
        metadata_dict.pop("id", None)
        metadata_dict.pop("last_response", None)
        metadata_dict.pop("profile", None)
        metadata_dict.pop("research_interests", None)
        return metadata_dict


async def get_content_string_from_url_sql(url: str) -> Optional[str]:
    """
    Get content string from a URL in the SQL database.
    """
    async for session in get_session():
        result = await session.execute(
            select(ModelResearcher).where(ModelResearcher.url == url)
        )
        metadata = result.scalars().first()
        if metadata:
            return make_researcher_str(metadata)
        else:
            return None


def count_tokens(messages):
    """
    Count the number of tokens in a list of messages.
    """
    model = os.getenv("LLM_MODEL")
    encoding = tiktoken.encoding_for_model(model)
    return sum(len(encoding.encode(message.content)) for message in messages)


def trim_message_history(messages, max_tokens=128000):
    """
    Trim the message history to fit within the token limit.
    """
    return trim_messages(
        messages, max_tokens=max_tokens, token_counter=count_tokens
    )
