# %%
# webloaders.py

from uos_grants.connectors.db import get_session
from uos_grants.connectors.models import StaffUrls as ModelStaffUrls
import asyncio
import re
from typing import Dict, List, Optional, Set
import aiohttp
from bs4 import BeautifulSoup, element
from langchain_core.utils.html import extract_sub_links
import logging
from datetime import datetime
import os
from contextlib import aclosing
from sqlalchemy import select

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


# %%
class AcademicDeptScraper:
    def __init__(
        self,
        url: str,
        containers: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        continue_on_failure: bool = True,
        use_async: bool = True,
        max_depth: int = 50,
        timeout: int = 10,
        base_url: str = "https://www.sheffield.ac.uk",
        remain_in_domain: bool = True,
        headers: Optional[Dict[str, str]] = None,
        max_concurrent_tasks: int = 10,
    ):
        self.url = url
        self.db_session_fn = get_session
        self.containers: Optional[List[str]] = containers or ["staff-profile-listing"]
        self.exclude_patterns: Optional[List[str]] = exclude_patterns
        self.continue_on_failure: bool = continue_on_failure
        self.use_async: bool = use_async
        self.max_depth: int = max_depth
        self.timeout: int = timeout
        self.base_url: str = base_url
        self.remain_in_domain: bool = remain_in_domain
        self.headers: Dict[str, str] = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        self.max_concurrent_tasks: int = max_concurrent_tasks

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _get_single_container(self, html: str, class_: str) -> element.Tag:
        """
        Get a single container from the HTML.

        Args:
            html (str): HTML content
            class_ (str): HTML class

        Returns:
            element.Tag: BeautifulSoup element
        """
        soup = BeautifulSoup(html, "lxml")
        container = soup.find(
            "div",
            class_=re.compile(class_),
        )
        if container:
            return container
        return None

    def _get_containers(self, html: str) -> Dict:
        """
        Get all containers from the HTML.

        Args:
            html (str): HTML content

        Returns:
            Dict: Dictionary of containers
        """
        containers = {}
        for container_type in self.containers:
            container = self._get_single_container(html, container_type)
            containers[container_type] = container
        return containers

    def _get_sub_links_from_container(self, container: element.Tag) -> Set[str]:
        """
        Get sub-links from a container

        Args:
            container (element.Tag): BeautifulSoup tag containing the HTML container

        Returns:
            Set[str]: Set of sub-links
        """
        links = set()
        for a in container.find_all("a", href=True):
            if not any(substr in a["href"] for substr in ["tel:", "mailto:", "#"]):
                link = a["href"]
                if self.exclude_patterns:
                    if any(
                        re.match(pattern, link) for pattern in self.exclude_patterns
                    ):
                        continue
                absolute_link = (
                    link
                    if link.startswith("http")
                    else f"https://www.sheffield.ac.uk{link}"
                )
                links.add(absolute_link)
        return links

    async def _async_get_child_links_recursive(
        self,
        url: str,
        visited: Set[str],
        *,
        session: Optional[aiohttp.ClientSession] = None,
        depth: int = 0,
    ) -> List[str]:
        """
        Get child links recursively and identify any relevant pages.

        Args:
            url (str): URL to fetch
            visited (Set[str]): Set of visited URLs
            session (Optional[aiohttp.ClientSession], optional): An active aiohttp session. Defaults to None.
            depth (int, optional): Maximum depth to scrape. Defaults to 0.

        Raises:
            ValueError: _description_
            e: _description_

        Returns:
            List[str]: _description_
        """
        if not self.use_async:
            raise ValueError("This method requires an async session.")

        if depth >= self.max_depth:
            return []

        close_session = session is None
        session = (
            session
            if session is not None
            else aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=True),
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=self.headers,
            )
        )
        visited.add(url)
        try:
            async with session.get(url) as response:
                if "text/html" not in response.headers["Content-Type"]:
                    return set(), ""
                text = await response.text()
        except aiohttp.ServerDisconnectedError:
            print(f"Server disconnected error for {url}")
            await asyncio.sleep(2)
            async with session.get(url) as response:
                if "text/html" not in response.headers["Content-Type"]:
                    return set(), ""
                text = await response.text()
        except (aiohttp.client_exceptions.InvalidURL, Exception) as e:
            if close_session:
                await session.close()
            if self.continue_on_failure:
                # print(f"Failed to fetch {url}: {e}")
                # print(f"{e.__class__.__name__}")
                return []
            else:
                raise e
        results = set()
        for container in self.containers:
            content = self._get_single_container(text, container)
            if content:
                links = self._get_sub_links_from_container(content)
                # self._write_to_file(
                #    links, f"uos_staff_links_depth_{self.max_depth}.txt"
                # )
                print(f"Found {len(links)} links in {url} at depth {depth}")
                await self._add_to_staff_url_db(links, session)
        if depth < self.max_depth - 1:
            sub_links = extract_sub_links(
                text,
                url,
                base_url=self.base_url,
                continue_on_failure=self.continue_on_failure,
            )
            if self.exclude_patterns:
                sub_links = [
                    link
                    for link in sub_links
                    if not any(keyword in link for keyword in self.exclude_patterns)
                ]
            if self.remain_in_domain:
                sub_links = [link for link in sub_links if self.base_url in link]
            sub_tasks = []
            to_visit = set(sub_links).difference(visited)
            for link in to_visit:
                sub_tasks.append(
                    self._async_get_child_links_recursive(
                        link, visited, session=session, depth=depth + 1
                    )
                )
            next_results = await asyncio.gather(*sub_tasks)
            for next_result in next_results:
                if len(next_result) == 2:
                    results_set, sub_result = next_result
                    if (
                        isinstance(sub_result, Exception)
                        or (results_set, sub_result) is None
                    ):
                        continue
                    for container in self.containers:
                        content = self._get_single_container(sub_result, container)
                        if content:
                            links = self._get_sub_links_from_container(content)
                            # self._write_to_file(
                            #     links,
                            #     f"uos_staff_links_depth_{self.max_depth}.txt",
                            # )
                            print(
                                f"Found {len(links)} links in {url} at depth {depth + 1}"
                            )
                            await self._add_to_staff_url_db(links, session)
        if close_session:
            await session.close()
        return results, text

    async def _add_to_staff_url_db(
        self,
        links: List[str],
        session: aiohttp.ClientSession,
    ):
        """
        Add URLs to the database.

        Args:
            links (List[str]): List of URLs
            session (aiohttp.ClientSession): Active aiohttp session
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        async def process_url(url: str):
            async with semaphore:
                try:
                    async with aclosing(self.db_session_fn()) as db_gen:
                        async for db in db_gen:
                            async with db.begin():
                                result = await db.execute(
                                    select(ModelStaffUrls).where(
                                        ModelStaffUrls.url == url
                                    )
                                )
                                existing = result.scalar_one_or_none()

                                if existing:
                                    logger.info(f"[UPDATE] {url}")
                                    existing.last_response = "200"
                                else:
                                    db.add(ModelStaffUrls(url=url, last_response="200"))
                                    logger.info(f"[INSERT] {url}")
                            break  # Only do one iteration of the generator
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout error for {url}")
                except aiohttp.ClientError as e:
                    logger.warning(f"Client error for {url}: {e}")
                except Exception as e:
                    logger.warning(f"Error for {url}: {e}")

        # Schedule all the URL tasks concurrently
        tasks = [asyncio.create_task(process_url(url)) for url in links]
        await asyncio.gather(*tasks)

    async def load(self) -> List[str]:
        """
        Load websites recursively from the starting URL.

        Returns:
            List[str]: List of URLs
        """
        visited = set()
        await self._async_get_child_links_recursive(self.url, visited)

    async def run(self):
        return await self.load()


# %%

scraper = AcademicDeptScraper(
    url="https://www.sheffield.ac.uk/departments/academic",
    containers=["staff-profile-listing", "stap-card"],
    exclude_patterns=["tel:", "mailto:", "#"],
    continue_on_failure=True,
    max_depth=100,
)
await scraper.run()
# %%
