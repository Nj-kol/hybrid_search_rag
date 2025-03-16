from typing import List
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import QueryResponse, Distance, VectorParams, models
from qdrant_client.http.models import Distance, VectorParams

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from fastembed import (
SparseTextEmbedding,
TextEmbedding,
LateInteractionTextEmbedding
)

class QdrantHybridRetriever(BaseRetriever):
    qdrant_host: str = Field(..., description="Qdrant DB hostname")
    qdrant_port: int = Field(..., description="Qdrant DB port")
    collection_name: str = Field(..., description="Name of the Qdrant collection")
    model_dir: str = Field(..., description="Directory containing the model")
    num_threads: int = Field(default=8, description="Number of threads to use")
    top_k: int = Field(default=5, description="Number of results to return") 
    
    # Model names
    sparse_vec_model_name: str = Field(default="Qdrant/bm25", description="Name of the sparse vector model")
    dense_vec_model_name: str = Field(default="intfloat/multilingual-e5-large", description="Name of the dense vector model")
    late_interaction_model_name: str = Field(default="colbert-ir/colbertv2.0", description="Name of the late interaction model")
    
    # Vector store configurations
    dense_vector_store: str = Field(default="dense", description="Name of dense vector store")
    dense_result_limit: int = Field(default=20, description="Number of results to return from dense search")
    sparse_vector_store: str = Field(default="bm25", description="Name of sparse vector store")

    ### Define default limits
    sparse_result_limit: int = Field(default=20,description="Number of results to return from sparse search")
    late_interaction_vector_store: str = Field(default="late_interaction",description="Name of late interaction vector store")
    hybrid_result_limit: int = Field(default=5,description="Number of results to return from hybrid search")
    
    # Model instances (initialized in __init__)
    sparse_embedding_model: SparseTextEmbedding = Field(default=None, description="Instance of sparse embedding model")
    dense_embedding_model: TextEmbedding = Field(default=None, description="Instance of dense embedding model")
    late_interaction_embedding_model: LateInteractionTextEmbedding = Field(default=None,description="Instance of late interaction model")

    class Config:
        arbitrary_types_allowed = True  # This is needed for the QdrantClient
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create your custom retriever
        # Manually add attributes that shouldn't be validated by Pydantic
        object.__setattr__(self, "vectordb_client", QdrantClient(self.qdrant_host, port=self.qdrant_port))

        # Initialize models
        self.sparse_embedding_model = SparseTextEmbedding(
            model_name=self.sparse_vec_model_name,
            cache_dir=self.model_dir,
            threads=self.num_threads
        )
        
        self.dense_embedding_model = TextEmbedding(
            model_name=self.dense_vec_model_name,
            cache_dir=self.model_dir,
            threads=self.num_threads
        )

        self.late_interaction_embedding_model = LateInteractionTextEmbedding(
            model_name=self.late_interaction_model_name,
            cache_dir=self.model_dir,
            threads=self.num_threads
        )
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:  
        sparse_query_vector = next(self.sparse_embedding_model.query_embed(query))
        dense_query_vector = next(self.dense_embedding_model.query_embed(query))
        late_query_vector = next(self.late_interaction_embedding_model.query_embed(query))
        sparse_vector_data = sparse_query_vector.as_object()
        sparse_vector = models.SparseVector(
            indices=sparse_vector_data['indices'],
            values=sparse_vector_data['values']
        )
        prefetch = [
            models.Prefetch(
                query=dense_query_vector,
                using=self.dense_vector_store,
                limit=self.dense_result_limit,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=self.sparse_vector_store,
                limit=self.sparse_result_limit
            ),
        ]
        query_response = self.vectordb_client.query_points(
                self.collection_name,
                prefetch=prefetch,
                query=late_query_vector,
                using=self.late_interaction_vector_store,
                with_payload=True,
                limit=self.hybrid_result_limit
        )
        documents = []
        for point in query_response.points:
            text = point.payload.get("text", "")  # Extract text content
            metadata= point.payload.get("metadata", "") 
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        return documents

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")