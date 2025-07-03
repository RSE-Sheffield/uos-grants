import asyncio
import logging
from pathlib import Path
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from uos_grants.scraper import SiteScraper
from uos_grants.graphBuilder import GraphSync
from uos_grants.graphEmbedder import run_graph_embedding


# ✅ Logging setup
log_dir = Path("/app/logs/scheduler")
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d")
log_file = log_dir / f"scheduler_{timestamp}.log"


# ✅ Custom handler that flushes immediately
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[FlushFileHandler(log_file, mode="a"), logging.StreamHandler()],
)

logger = logging.getLogger("uos_scheduler")


from neo4j import GraphDatabase
import os


def ensure_neo4j_indexes():
    logger.info("🔍 Checking Neo4j indexes...")

    indexes_to_ensure = [
        {
            "name": "department_index",
            "cypher": """
                CREATE VECTOR INDEX department_index
                FOR (d:Department)
                ON (d.embedding)
                OPTIONS {
                  indexConfig: {
                    `vector.dimensions`: 3072,
                    `vector.similarity_function`: 'cosine'
                  }
                }
            """,
        },
        {
            "name": "person_name_index",
            "cypher": """
                CREATE VECTOR INDEX person_name_index
                FOR (p:Person)
                ON (p.embedding)
                OPTIONS {
                  indexConfig: {
                    `vector.dimensions`: 3072,
                    `vector.similarity_function`: 'cosine'
                  }
                }
            """,
        },
        {
            "name": "research_interest_index",
            "cypher": """
                CREATE VECTOR INDEX research_interest_index
                FOR (ri:Research_Interest)
                ON (ri.embedding)
                OPTIONS {
                  indexConfig: {
                    `vector.dimensions`: 3072,
                    `vector.similarity_function`: 'cosine'
                  }
                }
            """,
        },
        {
            "name": "person_unique",
            "cypher": """
                CREATE CONSTRAINT person_unique IF NOT EXISTS
                FOR (p:Person)
                REQUIRE (p.name, p.url) IS UNIQUE
            """,
        },
    ]

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    try:
        with driver.session() as session:
            existing_indexes = session.run("SHOW INDEXES").data()
            existing_names = {row["name"] for row in existing_indexes}

            for index in indexes_to_ensure:
                if index["name"] not in existing_names:
                    logger.info(f"🆕 Creating missing index: {index['name']}")
                    session.run(index["cypher"])
                else:
                    logger.info(f"✅ Index exists: {index['name']}")

    finally:
        driver.close()

    logger.info("✅ Neo4j index check complete.")


# ✅ Async scrape + sync job
async def run_scrape_and_sync():
    logger.info("🚀 Starting scheduled scrape + graph sync + embedding job...")

    try:
        scraper = SiteScraper(sitemap_rate=5, scrape_rate=10)
        await scraper.run()
        logger.info("✅ Scrape job completed successfully.")
    except Exception as e:
        logger.exception(f"❌ Scrape job failed with error: {e}")

    try:
        graph_sync = GraphSync()
        await graph_sync.sync_graph_from_db()
        logger.info("✅ Graph sync job completed successfully.")
    except Exception as e:
        logger.exception(f"❌ Graph sync job failed with error: {e}")

    try:
        await run_graph_embedding()
    except Exception as e:
        logger.exception(f"❌ Graph embedding job failed with error: {e}")


# ✅ Main scheduler function (async)
async def main():
    scheduler = AsyncIOScheduler(timezone="Europe/London")

    # 🔥 Schedule for every Saturday at 00:00 (midnight)
    scheduler.add_job(
        run_scrape_and_sync,
        trigger=CronTrigger(day_of_week="sat", hour=0, minute=0),
        id="weekly_scrape_and_sync",
    )

    logger.info("📅 Scheduler started. Waiting for scheduled jobs...")

    scheduler.start()

    try:
        await asyncio.Event().wait()  # Run forever until interrupted
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Scheduler shutting down...")
        scheduler.shutdown()


# ✅ Startup sequence: run once then start scheduler
async def startup_sequence():
    logger.info("🚀 Running index check/migration step...")
    ensure_neo4j_indexes()

    logger.info("🚀 Running the initial scrape job...")
    await run_scrape_and_sync()

    logger.info("⏳ Starting the scheduler for future runs...")
    await main()


if __name__ == "__main__":
    asyncio.run(startup_sequence())
