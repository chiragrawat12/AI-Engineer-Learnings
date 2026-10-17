from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API KEY is not available")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b" 

def llm_ans(prompt):
    message = {
        "role": "user",
        "content": prompt
    }
    messages=[message]
    response = client.chat.completions.create(
        model=model, 
        messages=messages
    )
    return response.choices[0].message.content

bad_promt = """
 This is a User complain:
 My laptop is not working.
 Classify this.
"""

good_prompt = """
#ROLE:
You are a support assistant at a mobile/laptop company
#TASK
You have to classify the issue in a category
#CONSTRAINT
You have to classify the issue in one of three categories namely billing, technical, return.
#OUTPUT FORMAT
Your answer should be in one word only. The one word shoud be one of the categories given in constraints
#Example
For instance if a user compalin says he wants a refund then the category is Return
#FALLBACK
If the issue is unrelated to any of the categories mentioned in constraints, then the answer should be OTHER
This is a user complaint:
My Laptop is not working
"""

print(llm_ans(good_prompt))