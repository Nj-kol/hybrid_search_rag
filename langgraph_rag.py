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
        self.name = "PersonalGPT"
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
            print(f"User msg  is : {user_message}")
            ## Invoke
            metadata = body.get("metadata", {})
            user_id = metadata.get("user_id", "default_user_id")
            chat_id = metadata.get("chat_id", "default_chat_id")
            session_id = metadata.get("session_id", "default_session_id")
            print(f"chat_id, session_id: {chat_id}--{session_id}")
            config = {"configurable": {"thread_id": chat_id}}
            response = self.chat_service.invoke(user_message, config)
            return response
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise