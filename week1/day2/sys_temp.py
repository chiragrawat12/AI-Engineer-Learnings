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
content = "Suggest a name for my food company."
# content = "I love you baby!"
# message_system = {
#     "role": "system",
#     "content": "You are my loving girlfriend"
# }

# message_system = {
#     "role": "system",
#     "content": "You are my strict office colleague who is my manager also"
# }


message_system = {
    "role": "system",
    "content": "You are a brand manager who suggests name for my food company. Name should be in one word. Suggest one name only."
}

message = {
    "role": role,
    "content": content
}

messages = [message_system, message]

response = client.chat.completions.create(
    model=model, 
    messages=messages,
    temperature=2
)

print(response.choices[0].message.content)
