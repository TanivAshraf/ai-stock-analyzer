# --- generate_predictions.py (SELF-HEALING AUTO-DISCOVERY VERSION) ---

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
raw_gemini_key = os.environ.get('GEMINI_API_KEY', '')
GEMINI_API_KEY = raw_gemini_key.strip().strip('"').strip("'")

raw_news_key = os.environ.get('NEWS_API_KEY', '')
NEWS_API_KEY = raw_news_key.strip().strip('"').strip("'")

if not GEMINI_API_KEY:
    print("FATAL: GEMINI_API_KEY not found in environment secrets.")
    exit(1)

# --- Dynamic Model Auto-Discovery ---
def resolve_active_gemini_endpoint():
    """Queries Google's ListModels to dynamically pick the active Flash model."""
    preferred = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-flash']
    try:
        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        res = requests.get(list_url, timeout=10)
        if res.status_code == 200:
            models_data = res.json().get('models', [])
            supported = [
                m['name'].replace('models/', '')
                for m in models_data
                if 'generateContent' in m.get('supportedGenerationMethods', [])
            ]
            print(f"Discovered active models on account: {supported[:6]}")
            
            # Match preferred modern models first
            for candidate in preferred:
                if candidate in supported:
                    print(f"Selected model: {candidate}")
                    return f"https://generativelanguage.googleapis.com/v1beta/models/{candidate}:generateContent"
            
            # If none of preferred match, pick any available flash model
            for m_name in supported:
                if 'flash' in m_name:
                    print(f"Fallback selected flash model: {m_name}")
                    return f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent"
                    
            if supported:
                return f"https://generativelanguage.googleapis.com/v1beta/models/{supported[0]}:generateContent"
    except Exception as e:
        print(f"Model auto-discovery notice: {e}")

    # Fallback standard
    return "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

ACTIVE_API_URL = resolve_active_gemini_endpoint()

# --- Helper Functions ---
def get_stock_data_and_news(symbol):
    """Fetches historical stock data and recent news with robust column handling."""
    stock_data = yf.download(symbol, period="2mo", auto_adjust=True, progress=False)
    if stock_data.empty or len(stock_data) < 2:
        raise ValueError(f"yfinance returned insufficient data for {symbol}")
    
    if isinstance(stock_data.columns, pd.MultiIndex):
        try:
            close_series = stock_data['Close'][symbol]
        except KeyError:
            close_series = stock_data['Close'].iloc[:, 0]
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
    """Generates structured analysis using Gemini."""
    prompt = f"""
    You are an expert quantitative financial analyst. Analyze ticker {symbol}.
    Respond with a single, valid JSON object containing exactly these keys:
    - "sentiment": string, strictly one of "Bullish", "Bearish", or "Neutral".
    - "reasoning": string, concise 2-sentence rationale synthesizing price action and news.
    - "predicted_range": array of two numbers [predicted_low, predicted_high] for tomorrow's trading session.

    Historical 30-Day OHLCV Data:
    {historical_data.tail(30).to_string()}

    Recent 24-Hour News Headlines:
    {news_headlines}
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2
        }
    }

    target_url = f"{ACTIVE_API_URL}?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(target_url, headers=headers, json=payload, timeout=45)
            
            if response.status_code != 200:
                safe_err = response.text.replace(GEMINI_API_KEY, "[REDACTED]")
                print(f"API Error Response ({response.status_code}): {safe_err}")
                response.raise_for_status()
            
            res_json = response.json()
            candidates = res_json.get('candidates', [])
            if not candidates:
                raise ValueError("No candidate generation returned by the model.")
            
            raw_text = candidates[0]['content']['parts'][0]['text'].strip()
            clean_text = raw_text.replace('```json', '').replace('```', '').strip()
            parsed = json.loads(clean_text)
            
            sentiment = str(parsed.get('sentiment', 'Neutral')).capitalize()
            if sentiment not in ['Bullish', 'Bearish', 'Neutral']:
                sentiment = 'Neutral'
            parsed['sentiment'] = sentiment

            return parsed

        except Exception as e:
            safe_exception_str = str(e).replace(GEMINI_API_KEY, "[REDACTED]")
            print(f"Attempt {attempt + 1} for {symbol} failed: {safe_exception_str}")
            if attempt < max_retries - 1:
                time.sleep(4)
            else:
                raise RuntimeError(f"All retry attempts failed for {symbol}.")

def log_to_history_csv(log_data):
    """Appends daily evaluation row to history.csv cleanly."""
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

# --- Main Pipeline ---
def main():
    previous_predictions = {}
    try:
        with open(LIVE_JSON_FILE, 'r', encoding='utf-8') as f:
            prev_data = json.load(f)
            for item in prev_data.get('predictions', []):
                if 'symbol' in item and 'error' not in item:
                    previous_predictions[item['symbol']] = item
    except (FileNotFoundError, json.JSONDecodeError):
        print("Previous predictions file not found or empty. Initializing new run.")

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

            # 1. Update Live JSON
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

            # 2. Append to Persistent History CSV
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

            print(f"Successfully processed {symbol}: {ai_output.get('sentiment')} ({low_val} - {high_val})")

        except Exception as err:
            print(f"Error processing {symbol}: {err}")
            todays_data_for_json['predictions'].append({
                'symbol': symbol,
                'error': 'Data or prediction pipeline failure for this session.'
            })

    with open(LIVE_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(todays_data_for_json, f, indent=4)
        
    print(f"\nCompleted execution. Updated {LIVE_JSON_FILE} and appended to {HISTORY_CSV_FILE}.")

if __name__ == "__main__":
    main()
