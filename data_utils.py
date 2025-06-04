import requests
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import yfinance as yf

load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

finbert_model_name = "yiyanghkust/finbert-tone"
finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

def get_company_name_from_ticker(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName", ticker)
    except Exception:
        return ticker

def fetch_news(ticker):
    query_term = get_company_name_from_ticker(ticker)
    url = f"https://newsapi.org/v2/everything?q={query_term}&sortBy=publishedAt&language=en&apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        articles = response.json().get("articles", [])
        articles = [a for a in articles if all(ord(c) < 128 for c in a['title'])]
        return articles
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
