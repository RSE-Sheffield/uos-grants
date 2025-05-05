# %%
import os
from connectors.db import get_session
from connectors.repository import (
    create_tables,
)  # optional, if you need to ensure the table exists
from webloaders import AcademicDeptScraper, logger
from researchers import Researcher
import asyncio
from sqlalchemy import select
from connectors.db import get_session
from connectors.models import (
    Researcher as ModelResearcher,
    StaffUrls as ModelStaffUrls,
)
import chromadb
from uos_grants.embedding import BatchEmbeddingOpenAI
from itertools import islice
from connectors.repository import (
    insert_staff_url,
    update_researcher,
    insert_researcher,
    update_staff_url,
)
import requests
from contextlib import aclosing

from langchain_postgres import PGVector
from psycopg_pool import ConnectionPool

URLS = "../../uos_staff_links_depth_50.txt"

async def embed_researchers_to_pgvector(results, content_dict):
    collection_name = "sheffield_researchers"
    vector_db = PGVector(
        collection_name=collection_name,
        connection="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
        use_jsonb=True,
        embeddings=None,
    )
    ids = results["custom_id"].tolist()
    embeddings = results["embedding"].tolist()
    metadatas = results["metadata"].tolist()
    texts = [
        content_dict.get(url, "") for url in results["custom_id"].tolist()
    ]
    vector_db.add_embeddings(
        texts=texts,
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


def chunk_list(iterable, chunk_size=500):
    """Yield successive n-sized chunks from an iterable."""
    iterable = iter(iterable)
    while chunk := list(islice(iterable, chunk_size)):
        yield chunk


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


async def embed_researchers_to_chroma():
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    collection = chroma_client.get_or_create_collection(
        "uos_researcher_profiles"
    )

    content_dict = {}

    async for session in get_session():
        result = await session.execute(select(ModelResearcher))
        all_researchers = result.scalars().all()

    for batch_idx, researcher_batch in enumerate(
        chunk_list(all_researchers, 1000), start=1
    ):
        print(
            f"\n🔹 Processing Batch {batch_idx} ({len(researcher_batch)} researchers)"
        )
        batch = BatchEmbeddingOpenAI()

        for researcher in researcher_batch:
            metadata = {
                "url": researcher.url or "",
                "email": researcher.email or "",
                "name": researcher.name or "",
                "department": researcher.department or "",
                "main_role": researcher.main_role or "",
            }
            content = f"""{researcher.name if researcher.name else ""}
{researcher.department if researcher.department else ""}
{researcher.main_role if researcher.main_role else ""}
{researcher.profile if researcher.profile else ""}
{researcher.research_interests if researcher.research_interests else ""}"""
            content_dict[researcher.url] = content
            batch.add_content_string(content=content, metadata=metadata)

        batch.create_batch_job()
        results_list = []
        results = await batch.wait_for_batch_completion()
        results_list.append(results)
        for _, row in results.iterrows():
            url = row["metadata"].get("url")
            collection.upsert(
                embeddings=[row["embedding"]],
                metadatas=[row["metadata"]],
                ids=[row["custom_id"]],
                documents=[
                    content_dict.get(url, "")
                ],  # Fallback in case content key is missing
            )

        print(f"✅ Batch {batch_idx} completed and added to ChromaDB.")


# if __name__ == "__main__":
#     import asyncio
#     import nest_asyncio

#     nest_asyncio.apply()
#     asyncio.run(create_tables())

#     with open(URLS, "r") as file:
#         urls = file.readlines()
#         for url in urls:
#             asyncio.run(add_url_to_staff_db(url))

#     asyncio.run(add_staff_to_researcher_db())
#     # asyncio.run(main())

#     asyncio.run(embed_researchers_to_chroma())

# %%
