from langchain_ollama import ChatOllama

from apps.qdrant_hybrid_retriever import QdrantHybridRetriever
from apps.qdrant_rag_with_memory import ChatbotWithMemory

class QdrantRagService:
    def __init__(self):
        # Create your custom retriever
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        self.qdrant_collection_name="hybrid_search"
        self.embedddings_model_dir="/home/njkol/Models"
        self.embedddings_parallelism = 16
        
        ### Define different embedding model names
        self.sparse_vec_model_name = "Qdrant/bm25"
        self.dense_vec_model_name = "intfloat/multilingual-e5-large"
        self.late_interaction_model_name = "colbert-ir/colbertv2.0"
        ### Define Qdrant Vector store names
        self.sparse_vector_store_name = "bm25"
        self.dense_vector_store_name = "dense"
        self.late_interaction_vector_store_name = "late_interaction"
        ### Define Vector search result limits
        self.sparse_result_limit = 20
        self.dense_result_limit = 20
        self.hybrid_result_limit = 5
        
        ## Total documents to be returned by the Retriever
        self.top_k = 10

        self.retriever = QdrantHybridRetriever(
            qdrant_host=self.qdrant_host,
            qdrant_port=self.qdrant_port,
            collection_name=self.qdrant_collection_name,
            model_dir=self.embedddings_model_dir,
            num_threads=self.embedddings_parallelism,
            top_k=self.top_k,
            sparse_vec_model_name=self.sparse_vec_model_name,
            sparse_vector_store=self.sparse_vector_store_name,
            sparse_result_limit=self.sparse_result_limit,
            dense_vec_model_name=self.dense_vec_model_name,
            dense_vector_store=self.dense_vector_store_name,
            dense_result_limit=self.dense_result_limit,
            late_interaction_model_name=self.late_interaction_model_name,
            hybrid_result_limit=self.hybrid_result_limit,
            late_interaction_vector_store=self.late_interaction_vector_store_name
        )
    
        ## Initialize a chat model
        llm_name = "llama3.1:8b"
        self.llm = ChatOllama(
            base_url="http://localhost:11434",
            model=llm_name,
            temperature=0,
            num_predict=500,
            tfs_z=0.8,
            top_k=30,
            top_p=0.6
        )
        
        # Initialize the Chatbot
        self.chatbot = ChatbotWithMemory(self.llm, self.retriever)

    def invoke(self, query: str, config):
        return self.chatbot.query(query, config)

    def get_chat_session_history(self, config):
        return self.chatbot.get_history(config)