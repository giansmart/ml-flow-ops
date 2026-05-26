from openai import OpenAI
from dotenv import load_dotenv

import mlflow
from mlflow.genai import scorer
from mlflow.genai.scorers import Guidelines, Correctness

load_dotenv()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Evaluations")

mlflow.openai.autolog()

client = OpenAI()

def ask_agent(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer questions concisely in Spanish language in less than 10 words."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

eval_dataset = [
    {
        "inputs": {"question": "What is the capital of Peru?"},
        "expectations": {"expected_response": "Lima"}
    },
    {
        "inputs": {"question": "Who is the unique peruvian Writer who gained a Novel Prize"},
        "expectations": {"expected_response": "Mario Vargas Llosa"}
    },
     {
        "inputs": {"question": "Who is wrote Romeo and Juliet?"},
        "expectations": {"expected_response": "William Shakespeare"}
    }
]

@scorer
def is_concise(output: str) -> bool:
    """Evaluate if the response is concise (less than 10 words)"""
    return len(output.split()) <= 10

scorers = [Correctness(), Guidelines(name="is_spanish", guidelines="Answer must be provided in Spanish"), is_concise]

mlflow.genai.evaluate(data=eval_dataset, predict_fn=ask_agent, scorers=scorers)
