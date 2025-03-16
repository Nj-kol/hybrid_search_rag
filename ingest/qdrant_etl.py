import concurrent.futures
import time
import uuid
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, models
from qdrant_client.models import PointStruct
from fastembed import (
SparseTextEmbedding,
TextEmbedding,
LateInteractionTextEmbedding
)

class QdrantFastEmbedETLService:
    def __init__(self,
                 qdrant_host: str = "localhost", 
                 qdrant_port: int  = 6333,
                 embedding_models_dir: str = "/home/njkol/Models", 
                 sparse_vec_model_name = "Qdrant/bm25", 
                 dense_vec_model_name = "intfloat/multilingual-e5-large",
                 late_interaction_model_name ="colbert-ir/colbertv2.0",
                 embedding_threads = 16):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.embedding_models_dir = embedding_models_dir
        self.sparse_vec_model_name = sparse_vec_model_name
        self.dense_vec_model_name = dense_vec_model_name
        self.late_interaction_model_name = late_interaction_model_name
        self.embedding_threads = embedding_threads
        # Load models once
        self.sparse_embedding_model = SparseTextEmbedding(
            model_name=self.sparse_vec_model_name, 
            cache_dir=self.embedding_models_dir, 
            threads=self.embedding_threads
        )
        self.dense_embedding_model = TextEmbedding(
            model_name=dense_vec_model_name, 
            cache_dir=embedding_models_dir, 
            threads=embedding_threads
        )
        self.late_interaction_embedding_model = LateInteractionTextEmbedding(
            model_name=late_interaction_model_name, 
            cache_dir=embedding_models_dir, 
            threads=embedding_threads
        )

    def create_hybrid_search_collection(
        self,
        collection_name: str,
        dense_embedding_size: int  = 1024,
        late_interaction_embedding_size: int = 128
    ) -> None:
        """
        Ensures that a Qdrant collection exists. If it does not exist, creates it.
    
        :param client: QdrantClient instance
        :param collection_name: Name of the collection
        :param dense_embedding_size: Size of the dense vector dimension
        :param late_interaction_embedding_size: Size of the late interaction embedding vector dimension
        """
        client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        existing_collections = client.get_collections()
        collection_names = {col.name for col in existing_collections.collections}
        if collection_name not in collection_names:
            client.create_collection(
        	collection_name,
        	vectors_config={
        		"dense": models.VectorParams(
        			size=dense_embedding_size,
        			distance=models.Distance.COSINE,
        		),
        		"late_interaction": models.VectorParams(
        			size=late_interaction_embedding_size,
        			distance=models.Distance.COSINE,
        			multivector_config=models.MultiVectorConfig(
        				comparator=models.MultiVectorComparator.MAX_SIM,
        			),
                    hnsw_config=models.HnswConfigDiff(
                        m=0,  # Disable HNSW graph creation
                    )
        		),
        	},
        	sparse_vectors_config={
        		"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        	}
         )
            print(f"Collection '{collection_name}' created.")
        else:
            print(f"Collection '{collection_name}' already exists.")
        ## Close the underlying connection
        client.close()
            
    def create_multivector_points_from_documents(self, chunked_docs: List[Document]) -> List[PointStruct]:
        """
        Create Qdrant points from LangChain Documents and their embeddings.
        
        Args:
            dense_embeddings: Dense vector embeddings
            sparse_embeddings: BM25 sparse vectors
            late_interaction_embeddings: Late interaction embeddings
            documents: List of LangChain Document objects
        
        Returns:
            List of PointStruct ready for Qdrant insertion
        """
        # Function to generate embeddings within a process
        def embed_documents(embed_model, documents):
            return list(embed_model.embed([doc.page_content for doc in documents]))
        
        def process_embeddings(chunked_docs, dense_model, sparse_model, late_model, num_threads=4):
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:  # 3 threads for 3 embeddings
                dense_future = executor.submit(embed_documents, dense_model, chunked_docs)
                sparse_future = executor.submit(embed_documents, sparse_model, chunked_docs)
                late_future = executor.submit(embed_documents, late_model, chunked_docs)
        
                # Gather results
                dense_embeddings = dense_future.result()
                sparse_embeddings = sparse_future.result()
                late_interaction_embeddings = late_future.result()
        
            end_time = time.time()
            print(f"Embedding process completed in {end_time - start_time:.2f} seconds")
            return dense_embeddings, sparse_embeddings, late_interaction_embeddings

        # Assuming FastEmbed models are already instantiated
        dense_embeddings, sparse_embeddings, late_interaction_embeddings = process_embeddings(
            chunked_docs, self.dense_embedding_model, self.sparse_embedding_model, self.late_interaction_embedding_model
        )
        
        points = []
        for dense_emb, sparse_emb, late_emb, doc in zip(
            dense_embeddings,
            sparse_embeddings,
            late_interaction_embeddings,
            chunked_docs
        ):
            # Create point with UUID and document metadata
            point = PointStruct(
                id=str(uuid.uuid4()),  # Generate random UUID
                vector={
                    "dense": dense_emb,
                    "bm25": sparse_emb.as_object(),
                    "late_interaction": late_emb,
                },
                payload={
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
            )
            points.append(point)
        
        return points

    def upload_points(self,collection_name: str, 
                      points_batch: List[PointStruct],
                      threads: int = 4,
                      retries: int = 3):
        client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        start_time = time.time()  # Start timing
        client.upload_points(
            collection_name=collection_name,
            points=points_batch,
            parallel=threads,
            max_retries=retries,
        )
        end_time = time.time()  # End timing
        ## Close the underlying connection
        client.close()
        print(f"Uploading points completed in {end_time - start_time:.2f} seconds")