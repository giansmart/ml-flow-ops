import mlflow
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("OpenAI Prompts")

mlflow.openai.autolog()

client = OpenAI()
response = client.chat.completions.create(
    model="o4-mini",
    messages=[
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What's the weather like in Lima-Peru? Be brief"}
    ]
)

print(response)