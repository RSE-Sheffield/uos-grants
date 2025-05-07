import asyncio
import re
from typing import Dict, List, Optional, Set
from researchers import Researcher
import aiohttp
from bs4 import BeautifulSoup, element
from langchain_core.utils.html import extract_sub_links
from connectors.models import Researcher as ModelResearcher
from sqlalchemy import select
from connectors.repository import get_or_create_researcher, update_researcher
import logging
from datetime import datetime
import os

os.makedirs("./logs", exist_ok=True)

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"./logs/uos_scraping_job_{start_time}.log"
logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Only log WARNING and above by default
)
logger = logging.getLogger(__name__)

logger.info(f"Starting scraping job at {datetime.now()}")


class AcademicDeptScraper:
    def __init__(
        self,
        url: str,
        db_session_factory,
        max_depth: int = 2,
        timeout: int = 10,
        base_url: Optional[str] = None,
        headers: Optional[dict] = None,
        exclude_patterns: Optional[List[str]] = None,
        remain_in_domain: bool = True,
        continue_on_failure: bool = True,
        max_concurrent_tasks: Optional[int] = 3,
    ) -> None:
        self.url = url
        self.max_depth = max_depth
        self.timeout = timeout
        self.base_url = base_url or "https://www.sheffield.ac.uk"
        self.headers = headers
        self.exclude_patterns = exclude_patterns
        self.remain_in_domain = remain_in_domain
        self.continue_on_failure = continue_on_failure
        self.db_session_factory = db_session_factory
        self.semaphore = (
            asyncio.Semaphore(max_concurrent_tasks)
            if max_concurrent_tasks
            else None
        )
        self.temp_dict = {}

    async def _get_html(
        self, url: str, session: aiohttp.ClientSession
    ) -> Optional[str]:
        try:
            async with session.get(url) as response:
                if (
                    response.status == 200
                    and "text/html" in response.headers.get("Content-Type", "")
                ):
                    return await response.text()
                return str(response.status)
        except Exception as e:
            return str(e)

    def _get_staff_links(self, html: str) -> Set[str]:
        soup = BeautifulSoup(html, "lxml")
        container = soup.find(
            "div", class_=re.compile("staff-profile-listing")
        )
        links = set()
        if container:
            for a in container.find_all("a", href=True):
                href = a["href"]
                if any(substr in href for substr in ["tel:", "mailto:", "#"]):
                    continue
                if self.exclude_patterns and any(
                    re.search(pattern, href) for pattern in self.exclude_patterns
                ):
                    continue
                absolute_link = (
                    href
                    if href.startswith("http")
                    else f"{self.base_url}{href}"
                )
                links.add(absolute_link)
        return links

    async def _scrape_and_store_profile(
        self, url: str, session: aiohttp.ClientSession
    ):
        async def run():
            result = await self._get_html(url, session)

            if isinstance(result, str) and result.isdigit():
                async for db in self.db_session_factory():
                    async with db.begin():
                        existing = await get_or_create_researcher(
                            db, url, create_if_missing=False
                        )
                        if existing:
                            await update_researcher(
                                db, existing, {"last_response": result}
                            )
                return

            html = result

            if (
                'class="personinfo"' not in html
                and "class='personinfo'" not in html
            ):
                return

            researcher = await asyncio.to_thread(Researcher, html, url)

            data = researcher.to_dict()
            data["last_response"] = "200"

            required_fields = ["name", "url"]
            if not all(data.get(field) for field in required_fields):
                logger.warning(f"[SKIP] Incomplete data for {url}: {data}")
                return

            async for db in self.db_session_factory():
                async with db.begin():
                    result = await db.execute(
                        ModelResearcher.__table__.select().where(
                            ModelResearcher.url == url
                        )
                    )
                    existing = result.scalar_one_or_none()

                    if existing:
                        await update_researcher(db, existing, data)
                        logger.info(f"[UPDATE] {url}")
                    else:
                        db.add(ModelResearcher(**data))
                        logger.info(f"[INSERT] {url}")

        if self.semaphore:
            async with self.semaphore:
                await run()
        else:
            await run()

    async def _crawl(
        self,
        url: str,
        visited: Set[str],
        session: aiohttp.ClientSession,
        depth: int = 0,
    ):
        if depth >= self.max_depth or url in visited:
            return

        visited.add(url)
        html = await self._get_html(url, session)
        if not html or not isinstance(html, str) or not html.startswith("<"):
            return

        staff_links = self._get_staff_links(html)
        tasks = [
            self._scrape_and_store_profile(staff_url, session)
            for staff_url in staff_links
        ]
        await asyncio.gather(*tasks)

        if depth + 1 < self.max_depth:
            soup = BeautifulSoup(html, "lxml")
            links = [a["href"] for a in soup.find_all("a", href=True)]
            filtered_links = set()
            for link in links:
                if any(substr in link for substr in ["tel:", "mailto:", "#"]):
                    continue
                if self.exclude_patterns and any(
                    re.search(p, link) for p in self.exclude_patterns
                ):
                    continue
                absolute_link = (
                    link
                    if link.startswith("http")
                    else f"{self.base_url}{link}"
                )
                if (
                    self.remain_in_domain
                    and self.base_url not in absolute_link
                ):
                    continue
                filtered_links.add(absolute_link)

            sub_tasks = [
                self._crawl(link, visited, session, depth + 1)
                for link in filtered_links.difference(visited)
            ]
            await asyncio.gather(*sub_tasks)

    async def _refresh_existing_statuses(self, session: aiohttp.ClientSession):
        async for db in self.db_session_factory():
            async with db.begin():
                result = await db.execute(select(ModelResearcher.url))
                urls = [row[0] for row in result.fetchall()]

        async def check_and_update(url: str):
            status = await self._get_html(url, session)
            if isinstance(status, str) and status.isdigit():
                async for db in self.db_session_factory():
                    async with db.begin():
                        existing = await get_or_create_researcher(
                            db, url, create_if_missing=False
                        )
                        if existing:
                            await update_researcher(
                                db, existing, {"last_response": status}
                            )

        tasks = [check_and_update(url) for url in urls]
        await asyncio.gather(*tasks)

    async def run(self):
        visited = set()
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=True),
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self.headers,
        ) as session:
            await self._crawl(self.url, visited, session)
            await self._refresh_existing_statuses(session)

    def load(self):
        asyncio.run(self.run())

class AcademicDeptScraperFlex:
    def __init__(
        self,
        url: str,
        db_session_factory,
        max_depth: int = 2,
        timeout: int = 10,
        base_url: Optional[str] = None,
        headers: Optional[dict] = None,
        exclude_patterns: Optional[List[str]] = None,
        remain_in_domain: bool = True,
        continue_on_failure: bool = True,
        max_concurrent_tasks: Optional[int] = 3,
        containers: Optional[List[str]] = None,
    ) -> None:
        self.url = url
        self.max_depth = max_depth
        self.timeout = timeout
        self.base_url = base_url or "https://www.sheffield.ac.uk"
        self.headers = headers
        self.exclude_patterns = exclude_patterns
        self.remain_in_domain = remain_in_domain
        self.continue_on_failure = continue_on_failure
        self.db_session_factory = db_session_factory
        self.semaphore = (
            asyncio.Semaphore(max_concurrent_tasks)
            if max_concurrent_tasks
            else None
        )
        self.containers = containers or ["staff-profile-listing"]
        self.temp_dict = {}

    def _write_to_file(self, links: Set[str], filename: str):
        try:
            existing = set()
            try:
                with open(filename, "r") as f:
                    existing = set(f.read().splitlines())
            except FileNotFoundError:
                pass
            with open(filename, "a") as f:
                for link in links:
                    if link not in existing:
                        f.write(f"{link}\n")
        except Exception as e:
            print(f"Error writing to file {filename}: {e}")

    async def _get_html(
        self, url: str, session: aiohttp.ClientSession
    ) -> Optional[str]:
        try:
            async with session.get(url) as response:
                # if (
                #     response.status == 200
                #     #and "text/html" in response.headers.get("Content-Type", "")
                # ):
                text = await response.text()
                if "<html" not in text.lower():
                    logger.warning(f"[WARN] 200 OK but no HTML content at {url}")
                return text
                #return str(response.status)
        except Exception as e:
            #logger.warning(f"[ERROR] Failed to fetch {url}: {e}")
            return str(e)

    def _get_staff_links(self, html: str) -> Set[str]:
        soup = BeautifulSoup(html, "lxml")
        links = set()
        for container_type in self.containers:
            container = soup.find("div", class_=re.compile(container_type))
            if not container:
                continue
            for a in container.find_all("a", href=True):
                href = a["href"]
                if any(substr in href for substr in ["tel:", "mailto:", "#"]):
                    continue
                if self.exclude_patterns and any(
                    re.search(pattern, href) for pattern in self.exclude_patterns
                ):
                    continue
                absolute_link = (
                    href if href.startswith("http") else f"{self.base_url}{href}"
                )
                links.add(absolute_link)
        return links

    async def _scrape_and_store_profile(self, url: str, session: aiohttp.ClientSession):
        async def run():

            # if isinstance(result, str) and result.isdigit():
            #     async for db in self.db_session_factory():
            #         async with db.begin():
            #             existing = await get_or_create_researcher(
            #                 db, url, create_if_missing=False
            #             )
            #             if existing:
            #                 await update_researcher(
            #                     db, existing, {"last_response": result}
            #                 )
            #     return
            # if not isinstance(result, str) or "<html" not in result.lower():
            #     logger.warning(f"[SKIP] Got non-HTML result from {url}: {result[:100]}")
            #     self._write_to_file({url}, "missed_profiles.txt")
            #     return

            
            try:
                #result = await self._get_html(url, session)
                html = await self._get_html(url, session)
                self.temp_dict[url] = html
                researcher = await asyncio.to_thread(Researcher, self.temp_dict[url], url)
                data = researcher.to_dict()
                data["last_response"] = "200"
            except Exception as e:
                logger.warning(f"[ERROR] Failed to parse researcher at {url}: {e}")
                self._write_to_file({url}, "missed_profiles.txt")
                return

            if not all(data.get(field) for field in ["name", "url"]):
                logger.warning(f"[SKIP] Incomplete data for {url}: {data}")
                
                # ✅ New logging: Write raw HTML snippet to debug file
                with open("debug_incomplete_profiles.html", "a") as f:
                    f.write(f"<!-- {url} -->\n")
                    f.write(html)  # Only write the first 5k characters
                    f.write("\n\n")
                
                self._write_to_file({url}, "missed_profiles.txt")
                return

            async for db in self.db_session_factory():
                async with db.begin():
                    result = await db.execute(
                        ModelResearcher.__table__.select().where(
                            ModelResearcher.url == url
                        )
                    )
                    existing = result.scalar_one_or_none()

                    if existing:
                        await update_researcher(db, existing, data)
                        logger.info(f"[UPDATE] {url}")
                    else:
                        db.add(ModelResearcher(**data))
                        logger.info(f"[INSERT] {url}")
            # Remove the temporary HTML snippet from the dictionary
            del self.temp_dict[url]

        if self.semaphore:
            async with self.semaphore:
                await run()
        else:
            await run()

    async def _crawl(
        self,
        url: str,
        visited: Set[str],
        session: aiohttp.ClientSession,
        depth: int = 0,
    ):
        if depth >= self.max_depth or url in visited:
            return

        visited.add(url)
        html = await self._get_html(url, session)
        if not html or not isinstance(html, str) or not html.startswith("<"):
            return

        staff_links = self._get_staff_links(html)
        tasks = [
            self._scrape_and_store_profile(staff_url, session)
            for staff_url in staff_links
        ]
        await asyncio.gather(*tasks)

        if depth + 1 < self.max_depth:
            soup = BeautifulSoup(html, "lxml")
            links = [a["href"] for a in soup.find_all("a", href=True)]
            filtered_links = set()
            for link in links:
                if any(substr in link for substr in ["tel:", "mailto:", "#"]):
                    continue
                if self.exclude_patterns and any(
                    re.search(p, link) for p in self.exclude_patterns
                ):
                    continue
                absolute_link = (
                    link
                    if link.startswith("http")
                    else f"{self.base_url}{link}"
                )
                if (
                    self.remain_in_domain
                    and self.base_url not in absolute_link
                ):
                    continue
                filtered_links.add(absolute_link)

            sub_tasks = [
                self._crawl(link, visited, session, depth + 1)
                for link in filtered_links.difference(visited)
            ]
            await asyncio.gather(*sub_tasks)

    async def _refresh_existing_statuses(self, session: aiohttp.ClientSession):
        async for db in self.db_session_factory():
            async with db.begin():
                result = await db.execute(select(ModelResearcher.url))
                urls = [row[0] for row in result.fetchall()]

        async def check_and_update(url: str):
            status = await self._get_html(url, session)
            if isinstance(status, str) and status.isdigit():
                async for db in self.db_session_factory():
                    async with db.begin():
                        existing = await get_or_create_researcher(
                            db, url, create_if_missing=False
                        )
                        if existing:
                            await update_researcher(
                                db, existing, {"last_response": status}
                            )

        tasks = [check_and_update(url) for url in urls]
        await asyncio.gather(*tasks)

    async def run(self):
        visited = set()
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=True),
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self.headers,
        ) as session:
            await self._crawl(self.url, visited, session)
            await self._refresh_existing_statuses(session)

    def load(self):
        asyncio.run(self.run())