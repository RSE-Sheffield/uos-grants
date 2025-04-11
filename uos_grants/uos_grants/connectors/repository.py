# connectors/repository.py

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .models import Researcher, StaffUrls
from .db import engine, Base
from typing import Optional

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def insert_researcher(session: AsyncSession, data: dict) -> bool:
    researcher = Researcher(**data)
    session.add(researcher)
    try:
        await session.commit()
        return True
    except IntegrityError:
        await session.rollback()
        return False


async def get_researcher_by_email(session: AsyncSession, email: str) -> Researcher:
    result = await session.execute(
        select(Researcher).where(Researcher.email == email)
    )
    return result.scalar_one_or_none()

async def get_or_create_researcher(session: AsyncSession, url: str, create_if_missing=True) -> Optional[Researcher]:
    result = await session.execute(select(Researcher).where(Researcher.url == url))
    researcher = result.scalar_one_or_none()

    if researcher is None and create_if_missing:
        researcher = Researcher(url=url)
        session.add(researcher)
        await session.flush()

    return researcher


async def update_researcher(session: AsyncSession, researcher: Researcher, data: dict) -> None:
    for key, value in data.items():
        if not hasattr(researcher, key):
            continue

        if key == "url" and value != getattr(researcher, "url"):
            # Check if this URL is already used by another researcher
            result = await session.execute(
                select(Researcher).where(Researcher.url == value)
            )
            existing = result.scalar_one_or_none()
            if existing and existing.id != researcher.id:
                print(f"⚠️ URL '{value}' already exists for researcher ID {existing.id}. Skipping URL update.")
                continue

        setattr(researcher, key, value)

    await session.commit()

async def insert_staff_url(session: AsyncSession, url: str) -> bool:
    staff_url = StaffUrls(url=url, last_response="200")
    session.add(staff_url)
    try:
        await session.commit()
        return True
    except IntegrityError:
        await session.rollback()
        return False

async def get_staff_url(session: AsyncSession, url: str) -> Optional[StaffUrls]:
    result = await session.execute(
        select(StaffUrls).where(StaffUrls.url == url)
    )
    return result.scalar_one_or_none()

async def update_staff_url(session: AsyncSession, url: str, last_response: str) -> bool:
    staff_url = await get_staff_url(session, url)
    if staff_url:
        staff_url.last_response = last_response
        await session.commit()

    
