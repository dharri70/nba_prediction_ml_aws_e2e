import scrapy
from urllib.parse import urljoin
from scrapy.crawler import CrawlerProcess
from twisted.internet import reactor, defer
import logging
import re

# Configure logging
logging.basicConfig(
    filename='scraper.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://www.basketball-reference.com"
START_YEAR = 1977  # Only keep seasons from 1977 onward
YEAR_PATTERN = re.compile(r"(NBA|ABA)_(\d{4})\.html")  # Regex to match NBA and ABA seasons
OUTPUT_FILE = "links.txt"
BOXSCORE_FILE = "boxscore_urls.txt"

class LinksSpider(scrapy.Spider):
    name = "links_spider"
    custom_settings = {
        'DOWNLOAD_DELAY': 3.16,  # 19 requests per minute
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,  # Ensure only one request at a time
    }

    start_urls = [f"{BASE_URL}/leagues/"]

    def parse(self, response):
        logger.info("Parsing season links...")

        # Extract all potential season links
        links = response.xpath("//th[@scope='row' and contains(@class, 'left') and @data-stat='season']/a/@href").extract()
        filtered_links = []

        for link in links:
            match = YEAR_PATTERN.search(link)
            if match:
                year = int(match.group(2))  # Extract season year
                if year >= START_YEAR:
                    games_url = link.replace('.html', '_games.html')  # Modify link for game schedules
                    full_url = urljoin(BASE_URL, games_url)
                    filtered_links.append(full_url)

        if not filtered_links:
            logger.error("ERROR: No valid season links found. Check XPath structure.")

        # Write game URLs with monthly variations
        with open(OUTPUT_FILE, 'w') as file:
            buffer = []
            for link in filtered_links:
                for month in ['october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june']:
                    modified_url = link[:-5] + '-' + month + '.html'  # Append month to URL
                    buffer.append(modified_url + '\n')
                if len(buffer) > 1000:  # Write in batches of 1000 for efficiency
                    file.writelines(buffer)
                    buffer = []
            if buffer:  # Write remaining lines
                file.writelines(buffer)

        logger.info(f"Collected {len(filtered_links)} valid season links from {START_YEAR} onward.")
        logger.info(f"Links written to {OUTPUT_FILE}")

class BoxscoreSpider(scrapy.Spider):
    name = "boxscore_spider"
    custom_settings = {
        'DOWNLOAD_DELAY': 3.16,  # 19 requests per minute
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'AUTOTHROTTLE_ENABLED': False,  # Disable auto-throttle since we're setting a fixed delay
    }

    def __init__(self, *args, **kwargs):
        super(BoxscoreSpider, self).__init__(*args, **kwargs)
        self.backoff_factor = 1

    def start_requests(self):
        logger.info("Starting requests for box score links...")

        try:
            with open(OUTPUT_FILE, "r") as file:
                modified_urls = file.readlines()
        except FileNotFoundError:
            logger.error(f"File {OUTPUT_FILE} not found. Ensure the first spider ran successfully.")
            return

        for url in modified_urls:
            yield scrapy.Request(url.strip(), callback=self.parse)

    def parse(self, response):
        logger.info(f"Parsing Box Score Links from {response.url}")

        # Handle rate-limiting (429 Too Many Requests or 403 Forbidden)
        if response.status in [429, 403]:
            logger.warning(f"Received {response.status} status code. Increasing backoff factor.")
            self.backoff_factor *= 2  # Exponential backoff
            return defer.Deferred(lambda: self.crawler.engine.pause())

        # Extract box score links
        boxscore_urls = response.xpath("//a[contains(text(),'Box Score')]/@href").extract()

        # Write to file efficiently
        with open(BOXSCORE_FILE, "a") as file:
            buffer = [urljoin(BASE_URL, boxscore_url) + "\n" for boxscore_url in boxscore_urls]
            if buffer:
                file.writelines(buffer)

        # Adjust the delay for the next request based on backoff factor
        delay = self.custom_settings['DOWNLOAD_DELAY'] * self.backoff_factor
        defer.Deferred(lambda: self.crawler.engine.downloader.slots[response.url].delay(delay))

def start_scraping():
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    })
    
    process.crawl(LinksSpider)
    process.crawl(BoxscoreSpider)
    
    d = process.join()
    d.addBoth(lambda _: reactor.stop())
    reactor.run()

if __name__ == "__main__":
    start_scraping()
    logger.info("Scraping complete. Check links.txt and boxscore_urls.txt for results.")