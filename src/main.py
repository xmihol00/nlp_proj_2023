from typing import List

import pandas as pd

from src.dataset_preparation.pipeline import MovieScriptPreprocessor, SampleT
from src.scraping.dailyscript import DailyscriptScraper
from src.scraping.imsdb import ImsdbScraper

if __name__ == "__main__":
    scrapers = [
        ImsdbScraper(),
        DailyscriptScraper(),
        # Add more if needed
    ]

    # Run Scrapers
    for scraper in scrapers:
        scraper.run()

    # Apply preprocessing and merge datasources
    preprocessor = MovieScriptPreprocessor("./data/preprocessed_data.json")
    for scraper in scrapers:
        preprocessor.process_file(scraper.out_file)
        # For samples use:
        # preprocessor.process_sample(a_new_entry)

    # TODO: Continue here:
    # SampleT is just a type that somehow represents our samples (a dict)
    data: List[SampleT] = preprocessor.load_output()
    print(pd.DataFrame.from_records(data).head())
