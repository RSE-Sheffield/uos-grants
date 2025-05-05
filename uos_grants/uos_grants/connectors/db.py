# connectors/db.py

import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.ext.declarative import declarative_base

#INT_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@postgres:5432/uos_grants")
INT_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/uos_grants"

engine = create_async_engine(INT_DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

Base = declarative_base()

async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
