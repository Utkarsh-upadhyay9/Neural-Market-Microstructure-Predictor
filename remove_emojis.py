#!/usr/bin/env python3
import re
import os
import sys

def remove_emojis(text):
    """Remove emojis from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # various symbols
        ""  # specific emojis used
        "]+"
    )
    cleaned = emoji_pattern.sub('', text)
    # Clean up extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def clean_file(filepath):
    """Clean emojis from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned_content = remove_emojis(content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"Cleaned: {filepath}")
    except Exception as e:
        print(f"Error cleaning {filepath}: {e}")

def main():
    # Files to clean
    files_to_clean = [
        'README.md',
        'DEPLOYMENT.md',
        'config/deployment_config.yaml',
        'scripts/train_resume.py',
        '.env.template',
        'docker/Dockerfile.gpu'
    ]
    
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            clean_file(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()
