#%%
import asyncio
import os
from sqlalchemy import select, delete
from contextlib import aclosing
import aiohttp
import logging
from datetime import datetime
import random
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm.asyncio import tqdm as async_tqdm

from uos_grants.connectors.db import get_session
from uos_grants.connectors.models import StaffUrls as ModelStaffUrls


# ✅ Setup Logging to File
log_dir = "./logs/validation"
os.makedirs(log_dir, exist_ok=True)

log_filename = f"validation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_path = os.path.join(log_dir, log_filename)

logging.basicConfig(
    filename=log_path,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("uos_validator")


class Validator:
    def __init__(
        self,
        db_session_fn,
        timeout: int = 10,
        max_concurrent_tasks: int = 10,
        headers: dict = None,
    ):
        self.db_session_fn = db_session_fn
        self.timeout = timeout
        self.max_concurrent_tasks = max_concurrent_tasks
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (compatible; AcademicScraper/1.0)"
        }

        self.fetch_semaphore = asyncio.Semaphore(max_concurrent_tasks)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _check_url(self, url: str, session: aiohttp.ClientSession) -> bool:
        """Check if URL returns a valid status (200 or 3xx)."""
        async with self.fetch_semaphore:
            await asyncio.sleep(random.uniform(0.5, 1.5))  # Throttle to be polite

            try:
                async with session.get(url, allow_redirects=True) as response:
                    logger.info(f"[CHECK] {url} -> {response.status}")
                    return 200 <= response.status < 400
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"[ERROR] Failed checking {url}: {e}")
                return False

    async def _validate_url(self, url: str, session) -> None:
        """Independent DB session for each validation task."""
        is_valid = await self._check_url(url, session)

        async with aclosing(self.db_session_fn()) as db_gen:
            async for db in db_gen:
                try:
                    async with db.begin():
                        if is_valid:
                            logger.info(f"[VALID] {url}")
                        else:
                            logger.warning(f"[INVALID] {url} -> Removing")
                            await db.execute(
                                delete(ModelStaffUrls).where(ModelStaffUrls.url == url)
                            )
                except Exception as e:
                    logger.error(f"[DB ERROR] {url}: {e}")
                break  # Exit db_gen after one db instance

    async def run(self):
        # Get list of all URLs first
        urls = []
        async with aclosing(self.db_session_fn()) as db_gen:
            async for db in db_gen:
                result = await db.execute(select(ModelStaffUrls.url))
                urls = [row[0] for row in result.fetchall()]

        logger.info(f"[START] Found {len(urls)} links to validate")

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self.headers,
        ) as session:
            tasks = [
                self._validate_url(url, session)
                for url in urls
            ]

            for f in async_tqdm.as_completed(tasks, total=len(tasks), desc="Validating"):
                await f

        logger.info("[COMPLETE] Validation finished")


# ✅ Example usage
if __name__ == "__main__":
    validator = Validator(
        db_session_fn=get_session,
        timeout=10,
        max_concurrent_tasks=1000,
    )

    await validator.run()

# %%
