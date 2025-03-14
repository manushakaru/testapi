import json
import os

import requests
from common.utils.timing_logger import LOGGER, log_execution_time
from dotenv import load_dotenv

load_dotenv()


@log_execution_time
def get_search_results(
    query,
    num_results=10,
    num_pages=2,
    country_code="sg",
):
    url = "https://www.searchapi.io/api/v1/search"

    params = {
        "engine": "google",
        "q": query,
        "gl": country_code,
        "page": num_pages,
        "num": num_results,
        "api_key": os.getenv("SEARCH_API_KEY"),
    }

    response = requests.get(url, params=params)
    try:
        return response.json()
    except json.JSONDecodeError:
        print("Search API: get_search_results: Error decoding JSON response")
        LOGGER.error("Serper: get_search_results: Error decoding JSON response")
        return None


# url = "https://www.searchapi.io/api/v1/search"
# params = {
#     "engine": "google",
#     "q": "Do adults getting vaccinated for COVID-19?",
#     "gl": "sg",
#     "api_key": "3WEYDt7TedP96879z3JHcqVj",
# }

# response = requests.get(url, params=params)
# response = response.json()
# organic_results = response["organic_results"]
# print(organic_results)
