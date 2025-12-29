import os
from groq import Groq

key = os.environ.get("GROQ_API_KEY", "")
print("Key present:", bool(key), "Length:", len(key))

client = Groq(api_key=key)
resp = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Reply with just: OK"}],
    temperature=0
)
print(resp.choices[0].message.content)
