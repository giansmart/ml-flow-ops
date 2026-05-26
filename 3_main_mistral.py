import os
import mlflow
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("OpenAI Prompts")

# mlflow.openai.autolog()
# mlflow.langchain.autolog()
mlflow.mistral.autolog()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

chat_response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {"role": "user", "content": "What's the weather like in Lima-Peru? Be brief"}
    ]
)

print(chat_response)