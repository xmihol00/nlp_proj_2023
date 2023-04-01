import time
from tqdm import tqdm
import os
from typing import Optional
import urllib.error
import urllib.request
import json

from bs4 import BeautifulSoup

class GenreError(Exception):
    pass

class ScriptError(Exception):
    pass

class DailyscriptScraper:
    domain = "https://www.dailyscript.com/"
    file_name = "./data/scraped_dailyscript_data.json"

    def _get_soup(self, search_query):
        resp = urllib.request.urlopen(search_query)
        soup = BeautifulSoup(resp, "html.parser")
        return soup

    def _title_already_scraped(self, title):
        try:
            with open(self.file_name, mode="r+") as f:
                data = json.load(f)
                for d in data:
                    if d["title"] == title:
                        return True
            return False
        except FileNotFoundError:
            return False

    def _extract_movie_titles_and_script_links(self, url):
        soup = self._get_soup(url)
        tables = soup.find("body").find_all("table")
        result = []
        for table in tables:
            movie_list = table.find("ul")
            if movie_list is None:
                continue
            entries = movie_list.find_all("p")
            for entry in entries:
                reference = entry.find("a")
                result.append(
                    {
                        "title": reference.text,
                        "link": self.domain + reference.get("href"),
                    }
                )
        return result

    def _get_movie_scripts_and_genres(self):
        urls = [
            "https://www.dailyscript.com/movie.html",
            "https://www.dailyscript.com/movie_n-z.html",
        ]
        for url in urls:
            title_link_collection = (
                self._extract_movie_titles_and_script_links(url)
            )
            for item in tqdm(title_link_collection):
                title = item["title"]
                if self._title_already_scraped(title):
                    continue
                
                link = item["link"]
                if "html" not in link:
                    continue
                try:
                    genres = self._get_genres_from_first_imdb_result(title)
                    script = self._get_script(link)
                except GenreError:
                    print(f"Genres not found for {title}")
                    continue
                except ScriptError:
                    print(f"Script error for {title}")
                    continue

                yield title, script, genres

    def _get_script(self, url):
        soup = self._get_soup(url)
        body = soup.find("body")
        pre = soup.find("pre")
        if pre is not None:
            return pre.text
        elif body is not None:
            return body.text
        else:
            raise 


    def _get_genres_from_first_imdb_result(self, title: str):
        parsed_title = title.replace(" ", "+")
        search_query = (
            "https://www.imdb.com/search/"
            f"title/?title={parsed_title}&title_type=feature,tv_movie,short"
        )
        soup = self._get_soup(search_query)
        try:
            genres = soup.find("span", attrs={"class": "genre"}).text
        except AttributeError:
            raise GenreError
        genres = genres.replace("\n", "").replace(" ", "").split(",")
        return genres

    def _save_data(self, **kwargs):
        if not os.path.exists(self.file_name):
            with open(self.file_name, mode="w") as f:
                json.dump([kwargs], f)
        else:
            with open(self.file_name, mode="r+") as f:
                data = json.load(f)
                data.append(kwargs)
                f.seek(0)
                json.dump(data, f)

    def run(self):
        for title, script, genres in self._get_movie_scripts_and_genres():
            self._save_data(title=title, script=script, genres=genres)
            print("Wrote: ", title)


if __name__ == "__main__":
    scraper = DailyscriptScraper()
    scraper.run()
