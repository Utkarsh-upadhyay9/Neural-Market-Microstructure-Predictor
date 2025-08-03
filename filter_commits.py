#!/usr/bin/env python3
import sys
import re

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "🔥⚡📊🎯✅❌🚀💪🎉🔧🔄🌐📁💾🧠📈📅🏷️👤📧"
    "]+"
)

message = sys.stdin.read()
cleaned_message = emoji_pattern.sub('', message)
cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip()

# Ensure we have a meaningful commit message
if not cleaned_message or len(cleaned_message.strip()) < 5:
    cleaned_message = "Neural Market Predictor - Code Update"

print(cleaned_message)
