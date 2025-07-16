# --- generate_predictions.py (FINAL VERSION with Historical CSV Logging) ---

import os
import json
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import time
import csv # Import the CSV library

# --- Configuration ---
SYMBOLS = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']
LIVE_JSON_FILE = 'predictions.json' # File for the live dashboard
HISTORY_CSV_FILE = 'history.csv'   # New file for historical data logging

# --- API Setup ---
try:
    GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
except KeyError:
    print("FATAL: API keys not found in repository secrets.")
    exit(1)

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

# --- Helper Functions ---
def get_stock_data_and_news(symbol):
    """Fetches historical stock data and recent news."""
    stock_data = yf.download(symbol, period="2mo", auto_adjust=True)
    if stock_data.empty or len(stock_data) < 2:
        raise ValueError(f"yfinance returned insufficient data for {symbol}")
    news_headlines = "No recent news found."
    if NEWS_API_KEY:
        try:
            news_url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
            response = requests.get(news_url); response.raise_for_status()
            news_headlines = "\n".join([f"- {a['title']}" for a in response.json().get('articles', [])])
        except requests.RequestException as e: news_headlines = f"Could not fetch news: {e}"
    return stock_data, news_headlines

def get_ai_analysis(symbol, historical_data, news_headlines):
    """Generates structured analysis, retrying on failure."""
    prompt = f"""
    Analyze the financial data for **{symbol}**. Respond with a single, clean JSON object with keys: "sentiment", "reasoning", "predicted_low", "predicted_high".
    - "sentiment": Must be "Bullish", "Bearish", or "Neutral".
    - Do not include any text, markdown, or explanations outside of the JSON object.
    Historical Data: {historical_data.tail(30).to_string()}
    Recent News: {news_headlines}
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(GEMINI_API_URL, json=payload, timeout=45)
            response.raise_for_status()
            if not response.text: raise ValueError("Received empty response from API")
            json_response = response.json()
            candidates = json_response.get('candidates')
            if not candidates: raise ValueError(f"API response blocked or empty. Reason: {json_response.get('promptFeedback', 'Unknown')}")
            text_content = candidates[0]['content']['parts'][0]['text']
            clean_text = text_content.strip().replace('```json', '').replace('```', '')
            return json.loads(clean_text)
        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            print(f"Attempt {attempt + 1} for {symbol} failed: {e}")
            if attempt < max_retries - 1: time.sleep(2) 
            else: raise

# --- NEW: Function to log data to CSV ---
def log_to_history_csv(log_data):
    """Appends a new row of data to the history CSV file."""
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

# --- Main Execution Logic (UPGRADED) ---
def main():
    previous_predictions = {}
    try:
        with open(LIVE_JSON_FILE, 'r') as f:
            previous_data = json.load(f)
            for item in previous_data.get('predictions', []):
                if 'symbol' in item and 'error' not in item:
                    previous_predictions[item['symbol']] = item
    except (FileNotFoundError, json.JSONDecodeError):
        print("Previous prediction file not found. Starting fresh.")

    todays_data_for_json = {'last_updated': datetime.now(timezone.utc).isoformat(), 'predictions': []}

    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        try:
            if SYMBOLS.index(symbol) > 0:
                print("Waiting 5 seconds to respect API rate limits...")
                time.sleep(5)

            stock_data, news = get_stock_data_and_news(symbol)
            current_price = float(stock_data['Close'].iloc[-1])
            previous_close = float(stock_data['Close'].iloc[-2])
            
            ai_prediction_for_tomorrow = get_ai_analysis(symbol, stock_data, news)
            
            price_change = current_price - previous_close
            price_change_percent = (price_change / previous_close) * 100
            
            accuracy_check_hit = None
            yesterdays_predicted_range_str = "N/A"
            if symbol in previous_predictions:
                yesterdays_pred = previous_predictions[symbol]
                if 'predicted_range' in yesterdays_pred and yesterdays_pred['predicted_range'] and len(yesterdays_pred['predicted_range']) == 2:
                    pred_low = yesterdays_pred['predicted_range'][0]
                    pred_high = yesterdays_pred['predicted_range'][1]
                    if pred_low is not None and pred_high is not None:
                        accuracy_check_hit = pred_low <= current_price <= pred_high
                        yesterdays_predicted_range_str = f"${pred_low:.2f} - ${pred_high:.2f}"
            
            # 1. Prepare data for the LIVE JSON file
            live_record = {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'price_change': round(price_change, 2),
                'price_change_percent': round(price_change_percent, 2),
                'sentiment': ai_prediction_for_tomorrow.get('sentiment'),
                'reasoning': ai_prediction_for_tomorrow.get('reasoning'),
                'predicted_range': ai_prediction_for_tomorrow.get('predicted_range'),
                'accuracy_check': {
                    "yesterdays_predicted_range": yesterdays_predicted_range_str,
                    "todays_actual_price": f"${current_price:.2f}",
                    "hit": accuracy_check_hit
                } if accuracy_check_hit is not None else None
            }
            todays_data_for_json['predictions'].append(live_record)

            # 2. Prepare data for the HISTORY CSV file
            historical_log_record = {
                'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                'symbol': symbol,
                'actual_price': round(current_price, 4),
                'price_change': round(price_change, 4),
                'price_change_percent': round(price_change_percent, 4),
                'ai_sentiment_for_tomorrow': ai_prediction_for_tomorrow.get('sentiment'),
                'predicted_low_for_tomorrow': (ai_prediction_for_tomorrow.get('predicted_range') or [None, None])[0],
                'predicted_high_for_tomorrow': (ai_prediction_for_tomorrow.get('predicted_range') or [None, None])[1],
                'yesterdays_predicted_range': yesterdays_predicted_range_str,
                'accuracy_check_hit': accuracy_check_hit
            }
            log_to_history_csv(historical_log_record)

            print(f"Successfully processed and logged {symbol}.")

        except Exception as e:
            print(f"CRITICAL ERROR processing {symbol}: {e}")
            todays_data_for_json['predictions'].append({'symbol': symbol, 'error': str(e)})

    with open(LIVE_JSON_FILE, 'w') as f:
        json.dump(todays_data_for_json, f, indent=4)
    print(f"\nProcess complete. Updated {LIVE_JSON_FILE} and appended to {HISTORY_CSV_FILE}")

if __name__ == "__main__":
    main()