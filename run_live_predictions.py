#!/usr/bin/env python3
"""
Neural Market Predictor - Live Web Interface
Author: Utkarsh Upadhyay (@Utkarsh-upadhyay9)
Date: 2025-08-03 16:22:00 UTC
"""

import os
import sys
import subprocess
from datetime import datetime

print("🚀 Neural Market Predictor - Live Web Interface")
print("=" * 60)
print(f"Current Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"Author: Utkarsh Upadhyay (@Utkarsh-upadhyay9)")
print("=" * 60)

# Check if model exists
model_paths = [
    'models/extreme_heavy_final.keras',
    'models/extreme_heavy_model.keras'
]

model_found = False
for path in model_paths:
    if os.path.exists(path):
        print(f"✅ Found trained model: {path}")
        model_found = True
        break

if not model_found:
    print("ℹ️  No trained model found - using enhanced demo mode")

print("\n🌟 Features:")
print("  • Live stock search (100+ stocks)")
print("  • Trading recommendations (LONG/SHORT)")
print("  • Real-time updates (100ms live mode)")
print("  • Popular market indices")
print("  • Risk assessment & position sizing")
print("  • Entry/exit levels with stop-loss")
print("  • Personal watchlist")
print("  • Sector filtering")

print("\n🔧 Starting web server...")
print("📊 Web interface will be available at: http://localhost:8000")
print("=" * 60)

# Start the enhanced frontend
try:
    os.chdir('frontend/backend')
    if os.path.exists('server_enhanced.py'):
        subprocess.run([sys.executable, 'server_enhanced.py'])
    else:
        print("❌ Frontend server not found. Please run the setup commands first.")
except KeyboardInterrupt:
    print("\n\n👋 Server stopped by user")
except Exception as e:
    print(f"\n❌ Error starting server: {e}")
