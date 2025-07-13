# --- generate_predictions.py (UPGRADED VERSION) ---

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
try:
    GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
except KeyError:
    print("FATAL: API keys not found in repository secrets.")
    exit(1)

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

# --- Helper Functions (No changes needed here) ---
# ... (The get_stock_data_and_news and get_ai_analysis functions remain the same) ...
def get_stock_data_and_news(symbol):
    stock_data = yf.download(symbol, period="5d", auto_adjust=True) # Fetch 5 days to ensure we have previous day's data
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
    historical_data['SMA_20'] = historical_data['Close'].rolling(window=20).mean()
    prompt_data = historical_data.tail(30).to_string()
    prompt = f"""
    Analyze the financial data for **{symbol}**. Respond with a single, clean JSON object with keys: "sentiment", "reasoning", "predicted_low", "predicted_high".
    - "sentiment": Must be "Bullish", "Bearish", or "Neutral".
    - Do not include any text, markdown, or explanations outside of the JSON object.
    Historical Data: {prompt_data}
    Recent News: {news_headlines}
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(GEMINI_API_URL, json=payload); response.raise_for_status()
    text_content = response.json()['candidates'][0]['content']['parts'][0]['text']
    clean_text = text_content.strip().replace('```json', '').replace('```', '')
    return json.loads(clean_text)

# --- Main Execution Logic (UPGRADED) ---
def main():
    # Step 1: Read yesterday's predictions if the file exists
    previous_predictions = {}
    try:
        with open(OUTPUT_FILE, 'r') as f:
            previous_data = json.load(f)
            for item in previous_data.get('predictions', []):
                if 'symbol' in item and 'error' not in item:
                    previous_predictions[item['symbol']] = item
    except FileNotFoundError:
        print("Prediction file not found. Starting fresh.")
    except json.JSONDecodeError:
        print("Could not read old prediction file. Starting fresh.")

    # Step 2: Prepare container for today's new data
    todays_data = {
        'last_updated': datetime.utcnow().isoformat() + 'Z',
        'predictions': []
    }

    # Step 3: Process each symbol
    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        try:
            # Get fresh data
            stock_data, news = get_stock_data_and_news(symbol)
            
            # Extract today's and yesterday's prices
            current_price = float(stock_data['Close'].iloc[-1])
            previous_close = float(stock_data['Close'].iloc[-2])
            price_change = current_price - previous_close
            price_change_percent = (price_change / previous_close) * 100

            # Get new AI analysis for tomorrow
            ai_prediction_for_tomorrow = get_ai_analysis(symbol, stock_data, news)
            
            # --- ACCURACY CHECK LOGIC ---
            accuracy_check = None
            if symbol in previous_predictions:
                yesterdays_pred = previous_predictions[symbol]
                pred_low = yesterdays_pred['predicted_range'][0]
                pred_high = yesterdays_pred['predicted_range'][1]
                
                if pred_low is not None and pred_high is not None:
                    hit = pred_low <= current_price <= pred_high
                    accuracy_check = {
                        "yesterdays_predicted_range": f"${pred_low:.2f} - ${pred_high:.2f}",
                        "todays_actual_price": f"${current_price:.2f}",
                        "hit": hit
                    }
            
            # Assemble the new, richer data record
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
            print(f"ERROR processing {symbol}: {e}")
            todays_data['predictions'].append({'symbol': symbol, 'error': str(e)})

    # Step 4: Write the new data to the file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(todays_data, f, indent=4)
    print(f"\nSuccessfully generated rich predictions and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
