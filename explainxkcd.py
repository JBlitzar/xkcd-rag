import requests
from tqdm import tqdm
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class ExplainXKCDScraper:
    def __init__(self):
        self.base_url = "https://www.explainxkcd.com/wiki/api.php"
        self.cache_dir = "explainxkcd_cache"

    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def _make_request(self, params):
        response = requests.get(self.base_url, params=params)
        if response.status_code == 503:
            raise requests.exceptions.RequestException(f"Service unavailable (503)")
        return response

    def get_explanation(self, comic_number):
        # https://www.explainxkcd.com/wiki/api.php?action=parse&page=1&prop=wikitext&format=json
        params = {
            "action": "parse",
            "format": "json",
            "prop": "wikitext",
            "exintro": True,
            "page": str(comic_number),
            "redirects": 1,
        }
        response = self._make_request(params)
        if response.status_code != 200:
            print(f"HTTP {response.status_code} for comic {comic_number}")
            return "Explanation not found."

        try:
            data = response.json()
        except json.JSONDecodeError:
            print(f"Failed to decode JSON for comic {comic_number}")
            print(f"Response content: {response.text[:200]}")
            return "Explanation not found."

        pages = data.get("parse", {}).get("wikitext", {})
        if pages:
            return pages.get("*", "Explanation not found.")

        return "Explanation not found."

    def get_latest_comic_number(self):
        url = "https://xkcd.com/info.0.json"
        response = requests.get(url)
        data = response.json()
        print(data)
        return data.get("num", None)

    def get_all_explanations(self):
        latest_comic_number = self.get_latest_comic_number()
        explanations = {}
        if latest_comic_number is None:
            return explanations

        def fetch_explanation(comic_number):
            return comic_number, self.get_explanation(comic_number)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(fetch_explanation, comic_number): comic_number
                for comic_number in range(1, latest_comic_number + 1)
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Fetching explanations"
            ):
                comic_number, explanation = future.result()
                explanations[comic_number] = explanation

            return explanations

    def hydrate_and_refresh_cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        else:
            # exists, see if latest comic is cached
            latest_comic_number = self.get_latest_comic_number()
            cached_files = os.listdir(self.cache_dir)
            cached_comics = {
                int(f.split(".")[0]) for f in cached_files if f.endswith(".json")
            }
            if latest_comic_number in cached_comics:
                print("Cache is up to date.")
                return

        explanations = self.get_all_explanations()
        for comic_number, explanation in explanations.items():
            cache_file = os.path.join(self.cache_dir, f"{comic_number}.json")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"comic_number": comic_number, "explanation": explanation},
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

        self.load_cache()

    def load_cache(self):
        explanations = {}
        if not os.path.exists(self.cache_dir):
            return explanations

        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".json"):
                with open(
                    os.path.join(self.cache_dir, filename), "r", encoding="utf-8"
                ) as f:
                    data = json.load(f)
                    comic_number = data.get("comic_number")
                    explanation = data.get("explanation")
                    if comic_number and explanation:
                        explanations[comic_number] = explanation
        return explanations


if __name__ == "__main__":
    scraper = ExplainXKCDScraper()
    scraper.hydrate_and_refresh_cache()
