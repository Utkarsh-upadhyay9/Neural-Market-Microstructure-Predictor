"""
Environment variable loader
"""
import os
from pathlib import Path
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
        print("Environment variables loaded from .env")
    else:
        print("No .env file found, using system environment variables")

def get_api_key(service_name):
    """Get API key from environment variables"""
    key_mapping = {
        'alpha_vantage': 'ALPHA_VANTAGE_API_KEY',
        'newsapi': 'NEWSAPI_KEY',
        'nasdaq': 'NASDAQ_DATA_LINK_KEY'
    }
    
    env_var = key_mapping.get(service_name)
    if not env_var:
        raise ValueError(f"Unknown service: {service_name}")
    
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"API key not found for {service_name}. Set {env_var} in .env file")
    
    return api_key
