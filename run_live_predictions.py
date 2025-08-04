#!/usr/bin/env python3
"""
Neural Market Predictor - Live Trading System
Author: Utkarsh Upadhyay (@Utkarsh-upadhyay9)
Date: 2025-08-04 04:04:52 UTC
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from utils.env_loader import load_environment
    load_environment()
except ImportError:
    print("Loading environment variables...")
    from dotenv import load_dotenv
    load_dotenv()

def main():
    """Main entry point"""
    print("🚀 Neural Market Predictor - Live Trading System")
    print(f"📊 Author: Utkarsh Upadhyay (@Utkarsh-upadhyay9)")
    print(f"📅 Date: 2025-08-04 04:04:52 UTC")
    
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found!")
        print("💡 Copy .env.example to .env and add your API keys")
        return
    
    required_keys = ['ALPHA_VANTAGE_API_KEY', 'NEWSAPI_KEY', 'NASDAQ_DATA_LINK_KEY']
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing API keys: {', '.join(missing_keys)}")
        print("💡 Add these keys to your .env file")
        return
    
    print("✅ API keys loaded successfully")
    
    try:
        from frontend.backend.server import app
        print("🌐 Starting web server...")
        print("🔗 Access at: http://localhost:8000")
        
        app.run(host='0.0.0.0', port=8000, debug=False)
    except ImportError:
        print("❌ Server module not found. Starting basic version...")
        print("🔗 Access at: http://localhost:8000")
        
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        import webbrowser
        
        os.chdir('frontend')
        httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
        webbrowser.open('http://localhost:8000')
        httpd.serve_forever()

if __name__ == "__main__":
    main()
