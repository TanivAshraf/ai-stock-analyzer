# --- generate_predictions.py (SECURE & ENHANCED VERSION) ---

import os
import json
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import time
import csv

# --- Configuration ---
SYMBOLS = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']
LIVE_JSON_FILE = 'predictions.json'
HISTORY_CSV_FILE = 'history.csv'

# --- API Setup ---
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')

if not GEMINI_API_KEY:
    print("FATAL: GEMINI_API_KEY not found in environment secrets.")
    exit(1)

# Secure endpoint: No API key in the URL string
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# --- Helper Functions ---
def get_stock_data_and_news(symbol):
    """Fetches historical stock data and recent news."""
    stock_data = yf.download(symbol, period="2mo", auto_adjust=True, progress=False)
    if stock_data.empty or len(stock_data) < 2:
        raise ValueError(f"yfinance returned insufficient data for {symbol}")
    
    # Handle single or multi-index column structures cleanly
    if isinstance(stock_data.columns, pd.MultiIndex):
        close_series = stock_data['Close'][symbol]
    else:
        close_series = stock_data['Close']

    news_headlines = "No recent news found."
    if NEWS_API_KEY:
        try:
            news_url = (
                f"https://newsapi.org/v2/everything?q={symbol}&language=en"
                f"&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
            )
            res = requests.get(news_url, timeout=15)
            res.raise_for_status()
            articles = res.json().get('articles', [])
            if articles:
                news_headlines = "\n".join([f"- {a['title']}" for a in articles if a.get('title')])
        except Exception as e:
            news_headlines = "Could not fetch news headlines."

    return stock_data, close_series, news_headlines

def get_ai_analysis(symbol, historical_data, news_headlines):
    """Generates structured analysis from Gemini with native JSON mode."""
    prompt = f"""
    You are an expert financial analyst. Analyze the following data for ticker {symbol}.
    Return a strictly valid JSON object with the following keys:
    - "sentiment": Must be one of ["Bullish", "Bearish", "Neutral"].
    - "reasoning": A concise 2-sentence rationale synthesizing recent news and price momentum.
    - "predicted_range": A 2-element array of numbers representing [predicted_low, predicted_high] for tomorrow's session.

    Recent 30-Day Historical Data:
    {historical_data.tail(30).to_string()}

    Recent News Headlines:
    {news_headlines}
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.2
        }
    }

    # Pass API key securely via headers to prevent URL leakage in logs
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=45)
            response.raise_for_status()
            
            res_json = response.json()
            candidates = res_json.get('candidates', [])
            if not candidates:
                raise ValueError("Empty candidate response from Gemini API.")
            
            raw_text = candidates[0]['content']['parts'][0]['text'].strip()
            parsed = json.loads(raw_text)
            
            # Normalize and validate sentiment key
            sentiment = str(parsed.get('sentiment', 'Neutral')).capitalize()
            if sentiment not in ['Bullish', 'Bearish', 'Neutral']:
                sentiment = 'Neutral'
            parsed['sentiment'] = sentiment

            return parsed

        except Exception as e:
            print(f"Attempt {attempt + 1} for {symbol} failed.")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise RuntimeError(f"All retry attempts failed for {symbol}.")

def log_to_history_csv(log_data):
    """Appends a row to history.csv safely without duplication."""
    headers = [
        'date', 'symbol', 'actual_price', 'price_change', 
        'price_change_percent', 'ai_sentiment_for_tomorrow', 
        'predicted_low_for_tomorrow', 'predicted_high_for_tomorrow',
        'yesterdays_predicted_range', 'accuracy_check_hit'
    ]
    file_exists = os.path.isfile(HISTORY_CSV_FILE)
    
    with open(HISTORY_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)

# --- Main Execution Loop ---
def main():
    previous_predictions = {}
    try:
        with open(LIVE_JSON_FILE, 'r', encoding='utf-8') as f:
            prev_data = json.load(f)
            for item in prev_data.get('predictions', []):
                if 'symbol' in item and 'error' not in item:
                    previous_predictions[item['symbol']] = item
    except (FileNotFoundError, json.JSONDecodeError):
        print("Previous predictions file not found or empty. Initializing fresh run.")

    todays_data_for_json = {
        'last_updated': datetime.now(timezone.utc).isoformat(),
        'predictions': []
    }

    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        try:
            if SYMBOLS.index(symbol) > 0:
                time.sleep(4)

            stock_data, close_series, news = get_stock_data_and_news(symbol)
            current_price = float(close_series.iloc[-1])
            previous_close = float(close_series.iloc[-2])
            
            ai_output = get_ai_analysis(symbol, stock_data, news)
            
            price_change = current_price - previous_close
            price_change_percent = (price_change / previous_close) * 100
            
            # Historical accuracy range check
            accuracy_check_hit = None
            yesterdays_predicted_range_str = "N/A"
            if symbol in previous_predictions:
                yest_pred = previous_predictions[symbol]
                p_range = yest_pred.get('predicted_range')
                if p_range and len(p_range) == 2 and p_range[0] is not None and p_range[1] is not None:
                    low, high = float(p_range[0]), float(p_range[1])
                    accuracy_check_hit = (low <= current_price <= high)
                    yesterdays_predicted_range_str = f"${low:.2f} - ${high:.2f}"
            
            pred_range = ai_output.get('predicted_range') or [None, None]
            low_val = round(float(pred_range[0]), 2) if pred_range[0] is not None else None
            high_val = round(float(pred_range[1]), 2) if pred_range[1] is not None else None

            # 1. Dashboard JSON record
            live_record = {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'price_change': round(price_change, 2),
                'price_change_percent': round(price_change_percent, 2),
                'sentiment': ai_output.get('sentiment'),
                'reasoning': ai_output.get('reasoning'),
                'predicted_range': [low_val, high_val],
                'accuracy_check': {
                    "yesterdays_predicted_range": yesterdays_predicted_range_str,
                    "todays_actual_price": f"${current_price:.2f}",
                    "hit": accuracy_check_hit
                } if accuracy_check_hit is not None else None
            }
            todays_data_for_json['predictions'].append(live_record)

            # 2. History CSV record
            historical_log_record = {
                'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                'symbol': symbol,
                'actual_price': round(current_price, 4),
                'price_change': round(price_change, 4),
                'price_change_percent': round(price_change_percent, 4),
                'ai_sentiment_for_tomorrow': ai_output.get('sentiment'),
                'predicted_low_for_tomorrow': low_val,
                'predicted_high_for_tomorrow': high_val,
                'yesterdays_predicted_range': yesterdays_predicted_range_str,
                'accuracy_check_hit': accuracy_check_hit
            }
            log_to_history_csv(historical_log_record)

            print(f"Successfully processed {symbol}: {ai_output.get('sentiment')}")

        except Exception as err:
            print(f"Error processing {symbol}: {err}")
            # Sanitize error to avoid leaking system paths or credentials
            todays_data_for_json['predictions'].append({
                'symbol': symbol,
                'error': 'API or data retrieval failure for this trading session.'
            })

    with open(LIVE_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(todays_data_for_json, f, indent=4)
        
    print(f"\nCompleted run. Successfully updated {LIVE_JSON_FILE} and {HISTORY_CSV_FILE}.")

if __name__ == "__main__":
    main()
