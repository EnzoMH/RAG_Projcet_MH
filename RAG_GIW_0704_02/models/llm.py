from langchain_community.llms import HuggingFaceHub
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from app.config import HUGGINGFACEHUB_API_TOKEN

def get_llm():
    return HuggingFaceHub(
        repo_id="beomi/Llama-3-Open-Ko-8B",
        model_kwargs={"use_auth_token": HUGGINGFACEHUB_API_TOKEN},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )