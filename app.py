import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from data_utils import fetch_news, classify_sentiment

st.set_page_config(page_title="Stock Sentiment Analyzer", layout="centered")

st.title("Financial News Sentiment Analyzer (v0.3)")
st.subheader("Latest stock data with news sentiment analysis")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, MSFT):", value="AAPL").upper()

if st.button("Analyze"):
    if ticker:
        try:
            # Chart: Last 30 days only
            end_date = datetime.today().date()
            start_date = end_date - timedelta(days=30)

            st.info(f"Fetching data for: {ticker}")
            data = yf.download(ticker, start=start_date, end=end_date)

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if not data.empty:
                st.success(f"Data retrieved for {ticker}: {len(data)} records.")

                fig = px.line(
                    data,
                    x=data.index,
                    y="Close",
                    title=f"{ticker} Stock Price (Last 30 Days)",
                    labels={"Close": "Price (USD)", "index": "Date"},
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("Latest News & Sentiment")

                articles = fetch_news(ticker)

                if articles:
                    for article in articles:
                        title = article['headline']
                        summary = article.get('summary', 'No summary available.')
                        url = article.get('url', '#')

                        sentiment, confidence = classify_sentiment(title)

                        st.markdown(f"**{title}**")
                        st.write(summary)
                        st.write(f"Sentiment: {sentiment} ({confidence}% confidence)")
                        st.write(f"[Read more]({url})")
                        st.markdown("---")
                else:
                    st.info("No recent news articles found for this ticker.")

            else:
                st.warning("No stock data found for the past 30 days.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.warning("Please enter a valid ticker symbol.")
