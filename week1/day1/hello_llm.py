import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API KEY is not available")


client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"
role = "user"
content = "how are you?"

message = {
    "role": role,
    "content": content
}

messages = [message]

response = client.chat.completions.create(
    model=model, 
    messages=messages
)

print(response.choices[0].message.content)
