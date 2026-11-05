import os
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude! Write a brief greeting."}
    ]
)

print(response.content[0].text)