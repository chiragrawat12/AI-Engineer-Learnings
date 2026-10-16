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
content1 = "Hi!"
content2 = "Explain time travel in detail"
content3 = "provide 1000 word essay on machine learning"

contents = [content1, content2, content3]

for content in contents:
    message = {
        "role": role,
        "content": content
    }
    messages = [message]
    response = client.chat.completions.create(
        model=model, 
        messages=messages,
        max_tokens=5000
    )
    usage = response.usage
    print(f"prompt: {content} --> your tokens: {usage.prompt_tokens} completion tokens: {usage.completion_tokens} total tokens: {usage.prompt_tokens + usage.completion_tokens} finish reason: {response.choices[0].finish_reason}")
    print(response.choices[0].message.content)




