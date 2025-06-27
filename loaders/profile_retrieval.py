# %%
from typing import Optional, Set, List, Dict
from bs4 import BeautifulSoup
from yarl import URL
import aiohttp
import asyncio
import re

from uos_grants.connectors.db import get_session
from uos_grants.connectors.models import StaffUrls as ModelStaffUrls

import logging
from datetime import datetime
from contextlib import aclosing
from sqlalchemy import select

from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError
import random
import os

log_dir = "./logs/scraping"
os.makedirs(log_dir, exist_ok=True)

# Setup logging
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"./logs/scraping/uos_scraping_job_{start_time}.log"
logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("uos_scraper")
logger.info(f"Starting scraping job at {datetime.now()}")
logger.addHandler(logging.StreamHandler())


class AcademicDeptScraper:
    def __init__(
        self,
        url: str,
        db_session_fn,
        max_depth: int = 2,
        timeout: int = 10,
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        containers: Optional[List[str]] = None,
        continue_on_failure: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        remain_in_domain: bool = True,
        max_concurrent_tasks: int = 10,
    ) -> None:
        self.start_url = URL(url)
        self.base_url = URL(base_url) if base_url else self.start_url.origin()
        self.db_session_fn = db_session_fn

        self.max_depth = max_depth
        self.timeout = timeout
        self.headers = headers or {}
        self.containers = containers or [
            "stap-card",
            "staff-profile-list-block",
            "staff-profile-listing",
        ]
        self.continue_on_failure = continue_on_failure
        self.exclude_patterns = exclude_patterns or []
        self.remain_in_domain = remain_in_domain
        self.max_concurrent_tasks = max_concurrent_tasks

        self.fetch_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

    def is_in_domain(self, url: URL) -> bool:
        return url.host and url.host.endswith(self.base_url.host)

    def _sanitize_url(self, url: str) -> str:
        url_obj = URL(url).with_fragment(None)
        clean_url = str(url_obj).rstrip("/")
        return clean_url

    def _extract_all_links(self, html: str) -> Set[str]:
        """Extract all links on the page (for crawling)."""
        soup = BeautifulSoup(html, "lxml")
        links = set()

        for tag in soup.find_all("a", href=True):
            href = tag["href"]

            if any(substr in href for substr in ["tel:", "mailto:", "#"]):
                continue

            if any(re.search(pattern, href) for pattern in self.exclude_patterns):
                continue

            url = (
                URL(href) if href.startswith("http") else self.base_url.join(URL(href))
            )

            if self.remain_in_domain and not self.is_in_domain(url):
                continue

            clean_url = self._sanitize_url(str(url))
            links.add(clean_url)

        return links

    def _extract_links_from_containers(self, html: str) -> Set[str]:
        """Extract links only from the specified containers (for DB storage)."""
        soup = BeautifulSoup(html, "lxml")
        links = set()

        for container_class in self.containers:
            containers = soup.find_all(class_=re.compile(container_class))
            for container in containers:
                for tag in container.find_all("a", href=True):
                    href = tag["href"]

                    if any(substr in href for substr in ["tel:", "mailto:", "#"]):
                        continue

                    if any(re.search(pattern, href) for pattern in self.exclude_patterns):
                        continue

                    url = (
                        URL(href) if href.startswith("http") else self.base_url.join(URL(href))
                    )

                    if self.remain_in_domain and not self.is_in_domain(url):
                        continue

                    clean_url = self._sanitize_url(str(url))
                    links.add(clean_url)

        return links

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _fetch_html(
        self, url: URL, session: aiohttp.ClientSession
    ) -> Optional[str]:
        async with self.fetch_semaphore:
            await asyncio.sleep(random.uniform(0.5, 1.5))  # Throttle requests
            try:
                async with session.get(str(url)) as response:
                    logger.info(f"[HTTP] {url} returned {response.status}")
                    if "text/html" not in response.headers.get("Content-Type", ""):
                        return None
                    return await response.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"[ERROR] Failed fetching {url}: {e}")
                if self.continue_on_failure:
                    return None
                raise

    async def _scrape_recursive(
        self,
        url: URL,
        visited: Set[str],
        session: aiohttp.ClientSession,
        depth: int = 0,
    ) -> Set[str]:
        if depth >= self.max_depth or str(url) in visited:
            return set()

        visited.add(str(url))
        logger.info(f"[SCRAPE] Visiting {url} at depth {depth}")

        try:
            html = await self._fetch_html(url, session)
        except RetryError as e:
            logger.error(f"[RETRY FAILED] {url}: {e}")
            return set()

        if not html:
            return set()

        container_links = self._extract_links_from_containers(html)
        await self._store_links_in_db(container_links)

        crawl_links = self._extract_all_links(html)
        all_links = set(crawl_links)

        tasks = [
            self._scrape_recursive(URL(link), visited, session, depth + 1)
            for link in crawl_links - visited
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, set):
                all_links.update(result)

        return all_links

    async def _store_links_in_db(self, links: Set[str]) -> None:
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        async def process(link: str):
            async with semaphore:
                try:
                    async with aclosing(self.db_session_fn()) as db_gen:
                        async for db in db_gen:
                            async with db.begin():
                                result = await db.execute(
                                    select(ModelStaffUrls).where(
                                        ModelStaffUrls.url == link
                                    )
                                )
                                existing = result.scalar_one_or_none()

                                if existing:
                                    logger.info(f"[UPDATE] {link}")
                                    existing.last_response = "200"
                                else:
                                    db.add(
                                        ModelStaffUrls(url=link, last_response="200")
                                    )
                                    logger.info(f"[INSERT] {link}")
                            break
                except Exception as e:
                    logger.warning(f"[DB ERROR] {link}: {e}")

        await asyncio.gather(*(process(link) for link in links))

    async def load(self) -> Set[str]:
        visited: Set[str] = set()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self.headers,
        ) as session:
            results = await self._scrape_recursive(self.start_url, visited, session)
            logger.info(f"[COMPLETE] Scraped {len(results)} links")
            return results


# Example usage
headers = {
    "User-Agent": "Mozilla/5.0 (compatible; AcademicScraper/1.0)",
}

scraper = AcademicDeptScraper(
    url="https://www.sheffield.ac.uk/departments/academic",
    db_session_fn=get_session,
    max_depth=50,
    timeout=10,
    exclude_patterns=["publications", "attachment", ".pdf", "download"],
    headers=headers,
    max_concurrent_tasks=1000,
)

results = await scraper.load()

# %%
