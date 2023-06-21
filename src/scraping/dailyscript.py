from tqdm import tqdm
import os
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
    out_file = "./data/scraped_data/scraped_dailyscript_data.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    def _get_soup(self, search_query):
        resp = urllib.request.urlopen(search_query)
        soup = BeautifulSoup(resp, "html.parser")
        return soup # convert to DOM object

    def _title_already_scraped(self, title):
        try:
            with open(self.out_file, mode="r+") as f:
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
            for entry in entries: # find URLs to movie scripts
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
                link = item["link"]
                if self._title_already_scraped(title):
                    continue
                if "html" not in link:
                    continue
                try:
                    genres = self._get_genres_from_first_imdb_result(title)
                    script = self._get_script(link)
                    yield title, script, genres
                except GenreError:
                    print(f"Genres not found for {title}")
                    continue
                except ScriptError:
                    print(f"Script error for {title}")
                    continue
                except Exception as e:
                    print(e)
                    continue

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
        search_query = ( # search for movie title on IMDB
            "https://www.imdb.com/search/"
            f"title/?title={parsed_title}&title_type=feature,tv_movie,short"
        )
        soup = self._get_soup(search_query)
        try:
            genres = soup.find("span", attrs={"class": "genre"}).text # retrieve genres
        except AttributeError:
            raise GenreError
        genres = genres.replace("\n", "").replace(" ", "").split(",") # format the genres into a list
        return genres

    def _save_data(self, **kwargs):
        if not os.path.exists(self.out_file): # create file if it doesn't exist
            with open(self.out_file, mode="w") as f:
                json.dump([kwargs], f) 
        else: # append to file if it does exist
            with open(self.out_file, mode="r+") as f:
                data = json.load(f)
                data.append(kwargs)
                f.seek(0)
                json.dump(data, f)

    def run(self, overwrite=False):
        if overwrite or not os.path.exists(self.out_file):
            for title, script, genres in self._get_movie_scripts_and_genres():
                self._save_data(title=title, script=script, genres=genres)


if __name__ == "__main__":
    scraper = DailyscriptScraper()
    scraper.run(overwrite=True)
