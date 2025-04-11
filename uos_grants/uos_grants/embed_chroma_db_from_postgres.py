# %%
import asyncio
from sqlalchemy import select
from connectors.db import get_session
from connectors.models import Researcher as ModelResearcher
import chromadb
from uos_grants.embedding import BatchEmbeddingOpenAI
from itertools import islice


def chunk_list(iterable, chunk_size=500):
    """Yield successive n-sized chunks from an iterable."""
    iterable = iter(iterable)
    while chunk := list(islice(iterable, chunk_size)):
        yield chunk


async def embed_researchers_to_chroma():
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    collection = chroma_client.get_or_create_collection(
        "uos_researcher_profiles"
    )

    async for session in get_session():
        result = await session.execute(select(ModelResearcher))
        all_researchers = result.scalars().all()

    for batch_idx, researcher_batch in enumerate(
        chunk_list(all_researchers, 500), start=1
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
            print(content)
            batch.add_content_string(content=content, metadata=metadata)

        batch.create_batch_job()
        results = await batch.wait_for_batch_completion()

        for _, row in results.iterrows():
            collection.add(
                embeddings=row["embedding"],
                metadatas=row["metadata"],
                ids=row["custom_id"],
                documents=row.get(
                    "content", ""
                ),  # Fallback in case content key is missing
            )

        print(f"✅ Batch {batch_idx} completed and added to ChromaDB.")


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()
    asyncio.run(embed_researchers_to_chroma())

# %%
