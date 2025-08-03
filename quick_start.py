#!/usr/bin/env python3
"""
Quick start script to test everything with your actual API keys.
"""

import sys
import os
sys.path.append('.')

from test_all_apis import main as test_apis
from src.data.collector import DataCollector
import pandas as pd

def quick_test():
    """Quick test of all systems."""
    print("🚀 Neural Market Predictor - Quick Start")
    print("=" * 60)
    print("👤 User: Utkarsh-upadhyay9")
    print("📅 Date: 2025-08-02 16:47:24 UTC")
    print("🔑 APIs: Alpha Vantage, NewsAPI, Nasdaq Data Link")
    
    # Test APIs
    print("\n🧪 Testing API connections...")
    test_apis()
    
    # Quick data collection
    print("\n📊 Quick data collection test...")
    try:
        collector = DataCollector("config/config.yaml")
        
        # Get some Yahoo data (always works)
        data = collector.get_yahoo_data("AAPL", period="5d")
        
        if not data.empty:
            print(f"✅ Collected {len(data)} records for AAPL")
            print(f"📈 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            
            # Save sample
            os.makedirs("data/raw", exist_ok=True)
            collector.save_data(data, "data/raw/sample_aapl.csv")
            print("💾 Sample saved to data/raw/sample_aapl.csv")
        
        print("\n🎉 Quick start completed successfully!")
        print("🚀 Ready to build your market predictor!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_test()
