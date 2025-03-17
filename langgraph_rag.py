from typing import List, Union, Generator, Iterator
import os
from pydantic import BaseModel
from apps.qdrant_rag_service import QdrantRagService
import json

class Pipeline:

    class Valves(BaseModel):
        OLLAMA_BASE_URL: str
        VECTORSTORE_ENDPOINT: str

    def __init__(self):
        self.name = "AbhiGPT"
        self.valves = self.Valves(
            **{
                "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "VECTORSTORE_ENDPOINT": os.getenv("VECTORSTORE_ENDPOINT", "http://localhost:6333"),
            }
        )

    async def on_startup(self):
        # This function is called when the server is started.
        global chat_service, config
        self.chat_service = QdrantRagService()
        self.config = {"configurable": {"thread_id": "abc123"}}
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.
        try:
            if user_message.startswith("### Task:"):
                print(f"⚠️ Ignoring system message")
                return ""  # Return empty response or handle as needed
            print(f"Body is : {body}")
            print(f"User msg  is : {user_message}")
            ## Invoke
            thread_id = body.get("thread_id", f"session-{hash(user_message)}")
            config = {"configurable": {"thread_id": thread_id}}
            response = self.chat_service.invoke(user_message, self.config)
            return response
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise
