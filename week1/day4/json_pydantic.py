import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API KEY is not available")


client = Groq(api_key=my_api_key)

text = "Hello!, my name is Chirag Singh and my purchased iphone stopped working. My address is Limerick, Ireland. My email id is chiragrawat12@gmail.com. My contact number is 0894284845."

model = "openai/gpt-oss-120b"

#structure it
from pydantic import BaseModel 

class Ticket(BaseModel):
    name:str
    email:str
    contact_number:str
    address:str
    issue:str

schema = Ticket.model_json_schema()
response_format = {
    "type": "json_object"
}

system_prompt = f"""
Extract the Personal information from the ticket strictly based on this schema and give me a json output.
{schema}
"""
message_system = {
    "role": "system",
    "content": system_prompt
}

role = "user"
prompt = f'''
This is the customer ticket and please extract the customer details and provide it to me.
{text}
'''


message = {
    "role": role,
    "content": prompt
}
messages = [message_system, message]
response = client.chat.completions.create(
    model=model, 
    messages=messages,
    response_format=response_format
)
# usage = response.usage
# print(f"prompt: {prompt} --> your tokens: {usage.prompt_tokens} completion tokens: {usage.completion_tokens} total tokens: {usage.prompt_tokens + usage.completion_tokens} finish reason: {response.choices[0].finish_reason}")
answer = response.choices[0].message.content
print(answer)

import json
data_file = json.loads(answer)
ticket = Ticket(**data_file)

print(ticket.name)
print(ticket.address)
print(ticket.contact_number)
print(ticket.email)
print(ticket.issue)





