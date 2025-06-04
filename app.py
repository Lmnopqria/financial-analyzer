import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from data_utils import fetch_news, classify_sentiment

st.set_page_config(page_title="Stock Sentiment Analyzer", layout="centered")

st.title("Financial News Sentiment Analyzer (v0.2)")
st.subheader("Live stock data viewer with sentiment analysis")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, MSFT):", value="AAPL").upper()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", pd.to_datetime("2024-01-01"))
with col2:
    end_date = st.date_input("End date", pd.to_datetime("2024-12-31"))

if st.button("Fetch Stock Data"):
    if ticker:
        try:
            st.info(f"Fetching data for: {ticker}")
            data = yf.download(ticker, start=start_date, end=end_date)

            # Flatten columns if multi-index (e.g., [('Close', 'AAPL')])
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if not data.empty:
                st.success(f"Data retrieved for {ticker}: {len(data)} records.")

                fig = px.line(
                    data,
                    x=data.index,
                    y="Close",
                    title=f"{ticker} Stock Price",
                    labels={"Close": "Price (USD)", "index": "Date"},
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Fetch recent news for the ticker
                st.markdown("---")
                st.subheader(f"📰 Recent News Headlines About {ticker}")

                articles = fetch_news(ticker)

                if articles:
                    for article in articles[:5]:
                        title = article['title']
                        description = article['description'] or "No description available."
                        sentiment, confidence = classify_sentiment(title)

                        st.markdown(f"**{title}**")
                        st.write(description)
                        st.write(f"Sentiment: {sentiment} ({confidence}% confidence)")
                        st.write(f"[Read more]({article['url']})")
                        st.markdown("---")
                else:
                    st.info("No news articles found or API limit reached.")

            else:
                st.warning("No data found for this range.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.warning("Please enter a valid ticker symbol.")

