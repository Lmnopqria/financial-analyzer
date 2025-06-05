import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

load_dotenv()
API_KEY = os.getenv("FINNHUB_API_KEY")

finbert_model_name = "yiyanghkust/finbert-tone"
finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def fetch_news(ticker):
    today = datetime.now().date()
    thirty_days_ago = today - timedelta(days=30)
    from_date = thirty_days_ago.strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        articles = response.json()

        # Sort by timestamp (descending) to get most recent first
        articles.sort(key=lambda x: x.get("datetime", 0), reverse=True)

        # Keep only articles with valid titles (ASCII-only for English)
        articles = [
            a for a in articles
            if "headline" in a and all(ord(c) < 128 for c in a["headline"])
        ]

        return articles[:5]  # Return top 5 most recent
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

def classify_sentiment(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs).item()
        confidence = probs[0][predicted_class].item()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[predicted_class], round(confidence * 100, 2)
