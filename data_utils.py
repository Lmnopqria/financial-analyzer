import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

def fetch_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        articles = response.json().get("articles", [])

        # Filter to only include titles with ASCII characters (likely English)
        articles = [a for a in articles if all(ord(c) < 128 for c in a['title'])]

        return articles
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []
    