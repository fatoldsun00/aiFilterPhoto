# Class to load LLM from local or LMStudio

from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI

local_path = (
    "../langchain/models/mistral-7b-openorca.Q4_0.gguf"  # replace with your desired local file path
)

class LoadLLM:
    def __init__(self, isLMStudio, ModelPath=local_path):
        self.isLMStudio = isLMStudio
        self.ModelPath = ModelPath
        self.loadLLM()

    def loadLLM(self):
        if self.isLMStudio:
            self.llm = ChatOpenAI(
                openai_api_key="NULL",
                openai_api_base="http://localhost:1234/v1/",
                temperature=0,
            )
        else:
            self.llm = GPT4All(
                model=self.ModelPath,
                callbacks=[StreamingStdOutCallbackHandler()],
                verbose=True,
                temp=0,
            )
