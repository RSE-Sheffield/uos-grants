#%%
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from sqlalchemy import select, delete
from contextlib import aclosing
from datetime import datetime
from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError
from tqdm.asyncio import tqdm as async_tqdm
from pathlib import Path
import logging

from uos_grants.connectors.db import get_session
from uos_grants.connectors.models import Researcher
from uos_grants.researchers import Researcher as ResearcherScraper


class SiteScraper:
    def __init__(
        self,
        sitemap_url: str = "https://www.sheffield.ac.uk/sitemap.xml",
        sitemap_rate: int = 5,
        scrape_rate: int = 10,
    ):
        self.sitemap_url = sitemap_url
        self.sitemap_semaphore = asyncio.Semaphore(sitemap_rate)
        self.scrape_semaphore = asyncio.Semaphore(scrape_rate)

        # Logger setup
        log_dir = Path("/app/logs/link_collector")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_filename = f"link_collector_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        log_path = log_dir / log_filename

        logging.basicConfig(
            filename=log_path,
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.logger = logging.getLogger("uos_link_collector")

    # ✅ HTTP Fetch with Retry
    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def fetch(self, session, url):
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()

    # ✅ Recursive Sitemap Fetch
    async def fetch_sitemap_urls(self, session, sitemap_url):
        async with self.sitemap_semaphore:
            xml = await self.fetch(session, sitemap_url)
            soup = BeautifulSoup(xml, "xml")
            sitemap_tags = soup.find_all("sitemap")

            if sitemap_tags:
                sitemap_urls = [tag.find("loc").text for tag in sitemap_tags]
                tasks = [
                    self.fetch_sitemap_urls(session, sm_url)
                    for sm_url in sitemap_urls
                ]
                results = await asyncio.gather(*tasks)
                urls = [item for sublist in results for item in sublist]
                return urls
            else:
                url_tags = soup.find_all("url")
                urls = []
                for tag in url_tags:
                    loc = tag.find("loc").text
                    lastmod = tag.find("lastmod").text if tag.find("lastmod") else None
                    urls.append({"url": loc, "lastmod": lastmod})
                return urls

    async def collect_researcher_links(self):
        async with aiohttp.ClientSession() as session:
            sitemap_urls = await self.fetch_sitemap_urls(session, self.sitemap_url)

            people_links = [
                item for item in sitemap_urls if "/people/" in item["url"]
            ]

            print(f"Found {len(people_links)} people links.")
            self.logger.info(f"Found {len(people_links)} people links.")
            return people_links

    # ✅ DB Validation
    async def validate_db_against_sitemap(self, people_links):
        current_urls = set(item["url"] for item in people_links)

        async with aclosing(get_session()) as db_gen:
            async for db in db_gen:
                result = await db.execute(select(Researcher.url))
                db_urls = set(row[0] for row in result.all())

                urls_to_delete = db_urls - current_urls

                if urls_to_delete:
                    self.logger.info(f"Found {len(urls_to_delete)} obsolete entries to delete.")
                    print(f"Deleting {len(urls_to_delete)} obsolete entries...")

                    for url in urls_to_delete:
                        await db.execute(delete(Researcher).where(Researcher.url == url))
                        self.logger.info(f"[DELETE] {url}")

                    await db.commit()
                else:
                    self.logger.info("No obsolete entries found.")
                    print("No obsolete entries found.")

    # ✅ Filter for Updated or New Pages
    async def filter_updated_links(self, people_links):
        link_map = {item["url"]: item["lastmod"] for item in people_links}

        async with aclosing(get_session()) as db_gen:
            async for db in db_gen:
                result = await db.execute(
                    select(Researcher.url, Researcher.last_modified)
                    .where(Researcher.url.in_(link_map.keys()))
                )
                db_entries = result.all()

                db_map = {url: last_modified for url, last_modified in db_entries}

                to_process = []
                for url, lastmod in link_map.items():
                    db_lastmod = db_map.get(url)

                    if (db_lastmod is None) or (lastmod != db_lastmod):
                        to_process.append({"url": url, "lastmod": lastmod})

                skipped = len(people_links) - len(to_process)
                if skipped > 0:
                    print(f"Skipping {skipped} unchanged profiles.")
                    self.logger.info(f"Skipping {skipped} unchanged profiles.")

                return to_process

    # ✅ Fetch HTML with retries but no decorator so we can catch and act on errors directly
    async def fetch_html(self, session, url):
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()
        except aiohttp.ClientResponseError as e:
            raise e
        except Exception as e:
            raise e

    async def process_researcher(self, item, session):
        url = item["url"]
        lastmod = item["lastmod"]

        async with self.scrape_semaphore:
            async with aclosing(get_session()) as db_gen:
                async for db in db_gen:
                    try:
                        result = await db.execute(
                            select(Researcher).where(Researcher.url == url)
                        )
                        existing = result.scalar_one_or_none()

                        try:
                            html = await self.fetch_html(session, url)
                        except aiohttp.ClientResponseError as e:
                            if e.status in (404, 410):
                                await db.execute(delete(Researcher).where(Researcher.url == url))
                                await db.commit()
                                self.logger.warning(f"[ERROR DELETE] {url} — HTTP {e.status}")
                                return
                            else:
                                raise e  # Retry on server errors (e.g., 500)

                        scraper = ResearcherScraper(html, url)

                        now = datetime.utcnow()

                        data = {
                            "name": scraper.name,
                            "department": scraper.department,
                            "email": scraper.email,
                            "telephone": scraper.telephone,
                            "address": scraper.address,
                            "url": url,
                            "main_role": scraper.main_role,
                            "additional_roles": scraper.additional_roles,
                            "profile": scraper.profile,
                            "research_interests": scraper.research_interests,
                            "last_response": "200",
                            "last_modified": lastmod,
                            "last_scraped": now,
                        }

                        if existing:
                            self.logger.info(f"[UPDATE] {url}")
                            for key, value in data.items():
                                setattr(existing, key, value)
                            db.add(existing)
                        else:
                            self.logger.info(f"[INSERT] {url}")
                            db.add(Researcher(**data))

                        await db.commit()

                    except Exception as e:
                        self.logger.error(f"[ERROR] {url}: {e}")
                    break

    async def run(self):
        people_links = await self.collect_researcher_links()

        await self.validate_db_against_sitemap(people_links)

        to_process = await self.filter_updated_links(people_links)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "Mozilla/5.0 (compatible; LinkCollector/1.0)"},
        ) as session:
            tasks = [
                self.process_researcher(item, session)
                for item in to_process
            ]

            for f in async_tqdm.as_completed(tasks, total=len(tasks), desc="Processing"):
                await f
