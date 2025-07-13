# --- generate_predictions.py ---

import os
import json
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- Configuration ---
SYMBOLS = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']
OUTPUT_FILE = 'predictions.json'

# --- API Setup ---
# This script reads the keys securely from the GitHub repository secrets
try:
    GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
except KeyError:
    print("FATAL: API keys not found. Ensure GEMINI_API_KEY and NEWS_API_KEY are set in repository secrets.")
    exit(1)

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

# --- Core Functions ---

def get_stock_data_and_news(symbol):
    """Fetches historical stock data and recent news."""
    stock_data = yf.download(symbol, period="3mo", auto_adjust=True)
    if stock_data.empty:
        raise ValueError(f"yfinance returned no data for {symbol}")

    news_headlines = "No recent news found."
    if NEWS_API_KEY:
        try:
            news_url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
            response = requests.get(news_url)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            news_headlines = "\n".join([f"- {a['title']}" for a in articles])
        except requests.RequestException as e:
            news_headlines = f"Could not fetch news: {e}"
            
    return stock_data, news_headlines

def get_ai_analysis(symbol, historical_data, news_headlines):
    """Generates a structured qualitative analysis from the Gemini API."""
    historical_data['SMA_20'] = historical_data['Close'].rolling(window=20).mean()
    prompt_data = historical_data.tail(30).to_string()

    prompt = f"""
    Analyze the financial data for **{symbol}**.
    Based on the historical price data and recent news, provide a short-term forecast for the next trading day.
    Your response must be a single, clean JSON object with these exact keys: "sentiment", "reasoning", "predicted_low", "predicted_high".
    Do not include any text, markdown formatting like ```json, or explanations outside of the JSON object itself.

    **Historical Data:**
    {prompt_data}

    **Recent News Headlines:**
    {news_headlines}
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(GEMINI_API_URL, json=payload)
    response.raise_for_status()
    text_content = response.json()['candidates'][0]['content']['parts'][0]['text']
    clean_text = text_content.strip().replace('```json', '').replace('```', '')
    return json.loads(clean_text)

# --- Main Execution Logic ---

def main():
    """Main function to generate and save all predictions."""
    all_predictions = {
        'last_updated': datetime.utcnow().isoformat() + 'Z',
        'predictions': []
    }

    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        try:
            stock_data, news = get_stock_data_and_news(symbol)
            analysis = get_ai_analysis(symbol, stock_data, news)
            prediction_record = {
                'symbol': symbol,
                'current_price': round(float(stock_data['Close'].iloc[-1]), 2),
                'sentiment': analysis.get('sentiment'),
                'reasoning': analysis.get('reasoning'),
                'predicted_range': [analysis.get('predicted_low'), analysis.get('predicted_high')]
            }
            all_predictions['predictions'].append(prediction_record)
            print(f"Successfully processed {symbol}.")
        except Exception as e:
            print(f"ERROR processing {symbol}: {e}")
            all_predictions['predictions'].append({'symbol': symbol, 'error': str(e)})

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_predictions, f, indent=4)
    print(f"\nSuccessfully generated predictions and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
