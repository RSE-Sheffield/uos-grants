import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from uos_grants.scraper import SiteScraper


# ✅ Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger("uos_scheduler")


# ✅ Async scrape job
async def run_scraper():
    logger.info("🚀 Starting scheduled scrape job...")
    try:
        scraper = SiteScraper(
            sitemap_rate=5,
            scrape_rate=10,
        )
        await scraper.run()
        logger.info("✅ Scrape job completed successfully.")
    except Exception as e:
        logger.exception(f"❌ Scrape job failed with error: {e}")


# ✅ Main scheduler function (async)
async def main():
    scheduler = AsyncIOScheduler(timezone="Europe/London")

    # 🔥 Schedule for every Saturday at 00:00 (midnight)
    scheduler.add_job(
        run_scraper,  # directly the coroutine function, no lambda needed
        trigger=CronTrigger(day_of_week="sat", hour=0, minute=0),
        id="weekly_scrape",
    )

    logger.info("📅 Scheduler started. Waiting for scheduled jobs...")

    scheduler.start()

    try:
        await asyncio.Event().wait()  # Run forever until interrupted
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Scheduler shutting down...")
        scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(run_scraper())
    asyncio.run(main())
