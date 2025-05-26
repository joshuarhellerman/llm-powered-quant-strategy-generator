#!/usr/bin/env python3
import os
import anthropic

# Check environment variable
api_key = os.environ.get("ANTHROPIC_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'}")
if api_key:
    print(f"Key preview: {api_key[:8]}...{api_key[-4:]}")

# Test direct connection
try:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'API working' if you can read this."}]
    )
    print(f"✅ Claude API Response: {response.content[0].text}")
except Exception as e:
    print(f"❌ Claude API Error: {e}")