import os
import json
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
import time # We will use the time library for delays

# --- Configuration ---
SYMBOLS = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']
OUTPUT_FILE = 'predictions.json'

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

# --- UPGRADED, RESILIENT AI ANALYSIS FUNCTION ---
def get_ai_analysis(symbol, historical_data, news_headlines):
    """
    Generates structured analysis, retrying on failure and handling API instability.
    """
    prompt = f"""
    Analyze the financial data for **{symbol}**. Respond with a single, clean JSON object with keys: "sentiment", "reasoning", "predicted_low", "predicted_high".
    - "sentiment": Must be "Bullish", "Bearish", or "Neutral".
    - Do not include any text, markdown, or explanations outside of the JSON object.
    Historical Data: {historical_data.tail(30).to_string()}
    Recent News: {news_headlines}
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    # --- NEW: RETRY LOGIC ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(GEMINI_API_URL, json=payload, timeout=45)
            response.raise_for_status()
            
            # --- NEW: DEFENSIVE PARSING ---
            # Check for empty or malformed response before parsing JSON
            if not response.text:
                raise ValueError("Received empty response from API")

            json_response = response.json()
            candidates = json_response.get('candidates')
            if not candidates:
                 # This can happen if the model refuses to answer due to safety filters
                raise ValueError(f"API response blocked or empty. Reason: {json_response.get('promptFeedback', 'Unknown')}")
            
            text_content = candidates[0]['content']['parts'][0]['text']
            clean_text = text_content.strip().replace('```json', '').replace('```', '')
            
            # If we successfully parse the JSON, return it and exit the loop
            return json.loads(clean_text)

        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            print(f"Attempt {attempt + 1} for {symbol} failed: {e}")
            if attempt < max_retries - 1:
                # Wait before retrying (e.g., 2 seconds)
                time.sleep(2) 
            else:
                # If all retries fail, raise the final error
                raise

# --- Main Execution Logic ---
def main():
    # ... (The logic to read the previous predictions file remains the same) ...
    previous_predictions = {}
    try:
        with open(OUTPUT_FILE, 'r') as f:
            previous_data = json.load(f)
            for item in previous_data.get('predictions', []):
                if 'symbol' in item and 'error' not in item:
                    previous_predictions[item['symbol']] = item
    except (FileNotFoundError, json.JSONDecodeError):
        print("Previous prediction file not found or invalid. Starting fresh.")

    todays_data = {'last_updated': datetime.utcnow().isoformat() + 'Z', 'predictions': []}

    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        try:
            # --- NEW: ADDED A DELAY BETWEEN PROCESSING EACH SYMBOL ---
            if SYMBOLS.index(symbol) > 0:
                print("Waiting 5 seconds to respect API rate limits...")
                time.sleep(5) # Wait 5 seconds before processing the next stock

            stock_data, news = get_stock_data_and_news(symbol)
            current_price = float(stock_data['Close'].iloc[-1])
            previous_close = float(stock_data['Close'].iloc[-2])
            
            ai_prediction_for_tomorrow = get_ai_analysis(symbol, stock_data, news)
            
            # --- The rest of the logic remains the same ---
            price_change = current_price - previous_close
            price_change_percent = (price_change / previous_close) * 100
            
            accuracy_check = None
            if symbol in previous_predictions:
                if 'predicted_range' in yesterdays_pred and yesterdays_pred['predicted_range'] and len(yesterdays_pred['predicted_range']) == 2:
                    pred_low = yesterdays_pred['predicted_range'][0]
                    pred_high = yesterdays_pred['predicted_range'][1]
                    if pred_low is not None and pred_high is not None:
                        hit = pred_low <= current_price <= pred_high
                        accuracy_check = {"yesterdays_predicted_range": f"${pred_low:.2f} - ${pred_high:.2f}", "todays_actual_price": f"${current_price:.2f}", "hit": hit}
            
            prediction_record = {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'price_change': round(price_change, 2),
                'price_change_percent': round(price_change_percent, 2),
                'sentiment': ai_prediction_for_tomorrow.get('sentiment'),
                'reasoning': ai_prediction_for_tomorrow.get('reasoning'),
                'predicted_range': ai_prediction_for_tomorrow.get('predicted_range'),
                'accuracy_check': accuracy_check
            }
            todays_data['predictions'].append(prediction_record)
            print(f"Successfully processed {symbol}.")

        except Exception as e:
            print(f"CRITICAL ERROR processing {symbol}: {e}")
            todays_data['predictions'].append({'symbol': symbol, 'error': str(e)})

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(todays_data, f, indent=4)
    print(f"\nProcess complete. Saved predictions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
