#!/usr/bin/env python3
"""
Test script for all available APIs with your actual keys.
"""

import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json

# Your actual API keys
ALPHA_VANTAGE_KEY = "H8U3J9QXHBRSM1KU"
NEWSAPI_KEY = "799d9f41a7c240d18c06860ca1ffa31e"
NASDAQ_KEY = "i3KyqfeL5kfH2Cx6wSK2"

def test_alpha_vantage():
    """Test Alpha Vantage API."""
    print("🔍 Testing Alpha Vantage API...")
    
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': 'AAPL',
        'apikey': ALPHA_VANTAGE_KEY,
        'outputsize': 'compact'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if 'Time Series (Daily)' in data:
            print("✅ Alpha Vantage: SUCCESS")
            dates = list(data['Time Series (Daily)'].keys())
            latest_data = data['Time Series (Daily)'][dates[0]]
            print(f"📅 Latest date: {dates[0]}")
            print(f"💰 AAPL close: ${latest_data['4. close']}")
            return True
        else:
            print(f"❌ Alpha Vantage: {data}")
            return False
            
    except Exception as e:
        print(f"❌ Alpha Vantage error: {e}")
        return False

def test_yahoo_finance():
    """Test Yahoo Finance."""
    print("\n🔍 Testing Yahoo Finance...")
    
    try:
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="5d")
        
        if not data.empty:
            print("✅ Yahoo Finance: SUCCESS")
            print(f"📊 Got {len(data)} days of data")
            print(f"💰 Latest close: ${data['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ Yahoo Finance: No data")
            return False
            
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")
        return False

def test_newsapi():
    """Test NewsAPI."""
    print("\n🔍 Testing NewsAPI...")
    
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': 'Apple stock',
        'language': 'en',
        'sortBy': 'publishedAt',
        'apiKey': NEWSAPI_KEY,
        'pageSize': 5
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if data['status'] == 'ok' and data['articles']:
            print("✅ NewsAPI: SUCCESS")
            print(f"📰 Found {len(data['articles'])} articles")
            print(f"📅 Latest: {data['articles'][0]['title'][:50]}...")
            return True
        else:
            print(f"❌ NewsAPI: {data}")
            return False
            
    except Exception as e:
        print(f"❌ NewsAPI error: {e}")
        return False

def test_nasdaq_data_link():
    """Test Nasdaq Data Link."""
    print("\n🔍 Testing Nasdaq Data Link...")
    
    # Try a free dataset first
    url = f"https://data.nasdaq.com/api/v3/datasets/FRED/GDP/data.json"
    params = {
        'api_key': NASDAQ_KEY,
        'limit': 5
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if 'dataset_data' in data:
            print("✅ Nasdaq Data Link: SUCCESS")
            print(f"📊 Got {len(data['dataset_data']['data'])} data points")
            print(f"📅 Latest GDP data: {data['dataset_data']['data'][0]}")
            return True
        else:
            print(f"❌ Nasdaq Data Link: {data}")
            return False
            
    except Exception as e:
        print(f"❌ Nasdaq Data Link error: {e}")
        return False

def main():
    """Run all API tests."""
    print("🚀 Testing All APIs")
    print("=" * 50)
    print(f"⏰ Time: {datetime.now()}")
    print(f"👤 User: Utkarsh-upadhyay9")
    
    results = {
        'Alpha Vantage': test_alpha_vantage(),
        'Yahoo Finance': test_yahoo_finance(),
        'NewsAPI': test_newsapi(),
        'Nasdaq Data Link': test_nasdaq_data_link()
    }
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    
    working_apis = []
    for api, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {api}")
        if status:
            working_apis.append(api)
    
    print(f"\n🎉 {len(working_apis)}/{len(results)} APIs working!")
    
    if len(working_apis) >= 2:
        print("💡 You have enough APIs to start building your predictor!")
    else:
        print("⚠️  You may need to troubleshoot API connections.")

if __name__ == "__main__":
    main()
