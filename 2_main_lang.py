import mlflow
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

load_dotenv()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("OpenAI Prompts")

# mlflow.openai.autolog()
mlflow.langchain.autolog()

@tool
def get_weather(city: str) -> str:
    """Tool to get a very basic description about the wheater of a specific city"""
    return f"he wheater in {city} is sunny and 21 Celcius"


chat = ChatOpenAI(model="o4-mini").bind_tools([get_weather])
response = chat.invoke(
    [
        SystemMessage(content="You are a helpful weather assistant."),
        HumanMessage(content="What's the weather like in Lima-Peru? Be brief"),
    ]
)

print(response)