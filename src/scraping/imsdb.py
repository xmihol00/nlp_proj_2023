import re
import os
import urllib.request
from typing import List
from bs4 import BeautifulSoup

from scrapy.crawler import CrawlerProcess
from w3lib.url import url_query_cleaner
from scrapy.spiders.crawl import CrawlSpider, LinkExtractor, Rule


def process_links(links):
    for link in links:
        link.url = url_query_cleaner(link.url)
        yield link


class ImsdbSpider(CrawlSpider):
    name = "imsdb"
    base_url = "https://imsdb.com"
    allowed_domains = ["imsdb.com"]
    start_urls = ["https://imsdb.com/all-scripts.html"]
    rules = (
        Rule(
            LinkExtractor(
                allow=re.escape("https://imsdb.com/Movie%20Scripts/")
            ),
            callback="parse_item", # callback function to parse the page
        ),
    )
    custom_settings = {
        "ROBOTSTXT_OBEY": False,   # ignore robots.txt
    }

    def _get_genres_from_links(self, links) -> List[str]:
        genres = []
        for link in links:
            ref = link.get("href")
            if "genre" in ref:  # locate the correct link
                genres.append(link.text)
        return genres

    def _get_script_from_links(self, links):
        script_link = None
        for link in links:
            if "Read" in link.text and "Script" in link.text:  # locate the correct link
                script_link = self.base_url + link.get("href")
        response = urllib.request.urlopen(script_link)
        soup = BeautifulSoup(response, "html.parser")
        return str(soup.find("td", {"class": "scrtext"}).find("pre").contents)

    def parse_item(self, response):
        soup = BeautifulSoup(response.text) # convert to DOM object
        imdbs_box = soup.find(text="Genres").parent.parent
        links = imdbs_box.find_all("a")
        title = (
            response.url.split("/")[-1]
            .replace("%20Script.html", "")
            .replace("%20", " ")
        )
        genres = self._get_genres_from_links(links)
        script = self._get_script_from_links(links)
        yield {"title": title, "genres": genres, "script": script} # return newly scraped movie script


class ImsdbScraper():
    out_file = "./data/scraped_data/scraped_imsdb_data.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    def run(self, overwrite=False):
        if overwrite or not os.path.exists(self.out_file):
            process = CrawlerProcess(
                settings={
                    "FEEDS": {
                        "./data/scraped_data/scraped_imsdb_data.json": {"format": "json"},
                    },
                }
            )
            process.crawl(ImsdbSpider)
            process.start()


if __name__ == "__main__":
    ImsdbScraper().run(overwrite=True)
    with open("./data/scraped_data/scraped_imsdb_data.json", "r") as f:
        data = f.read()
        data = re.sub(r"^\]\[$", ",", data) # fix json formatting of multiple array objects into a single array
    
    with open("./data/scraped_data/scraped_imsdb_data.json", "w") as f:
        f.write(data)
