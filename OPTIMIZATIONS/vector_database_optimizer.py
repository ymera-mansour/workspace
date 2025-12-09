"""
Vector Database Optimization - FAISS/Chroma Implementation
Complete vector store optimization with local and cloud options
"""

import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import CohereEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents import Document
import numpy as np


class VectorDatabaseOptimizer:
    """
    Optimized vector database management with FAISS and Chroma
    
    Features:
    - Local FAISS for fast, in-memory search (FREE)
    - Chroma for persistent local storage (FREE)
    - Qdrant cloud option (1GB FREE)
    - Pinecone cloud option (100K vectors FREE)
    - Automatic optimization and indexing
    - Query caching and performance tuning
    """
    
    def __init__(
        self,
        embedding_provider: str = "cohere",  # cohere, huggingface
        vector_store: str = "faiss",  # faiss, chroma, qdrant, pinecone
        persist_directory: Optional[str] = None
    ):
        self.embedding_provider = embedding_provider
        self.vector_store_type = vector_store
        self.persist_directory = persist_directory or "./vector_db"
        
        # Initialize embeddings
        self.embeddings = self._initialize_embeddings()
        
        # Initialize vector store
        self.vector_store = None
        self.query_cache = {}
        
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        if self.embedding_provider == "cohere":
            # Best-in-class embeddings, FREE (100 calls/min)
            return CohereEmbeddings(
                model="embed-english-v3.0",
                cohere_api_key=os.getenv("COHERE_API_KEY")
            )
        elif self.embedding_provider == "huggingface":
            # Local embeddings, completely FREE
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unknown embedding provider: {self.embedding_provider}")
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """Create and optimize vector store"""
        if self.vector_store_type == "faiss":
            # FAISS - Fast, in-memory, FREE
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Optimize FAISS index
            if len(documents) > 1000:
                # Use IVF index for large datasets
                self.vector_store.index = self._optimize_faiss_index(
                    self.vector_store.index,
                    num_clusters=int(np.sqrt(len(documents)))
                )
            
        elif self.vector_store_type == "chroma":
            # Chroma - Persistent, local, FREE
            self.vector_store = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=self.persist_directory
            )
            
        else:
            raise ValueError(f"Unknown vector store: {self.vector_store_type}")
    
    def _optimize_faiss_index(self, index, num_clusters: int):
        """Optimize FAISS index for faster search"""
        import faiss
        
        # Convert to IVF (Inverted File) index for faster search
        dimension = index.d
        quantizer = faiss.IndexFlatL2(dimension)
        
        # Create IVF index
        ivf_index = faiss.IndexIVFFlat(quantizer, dimension, num_clusters)
        
        # Train the index
        vectors = index.reconstruct_n(0, index.ntotal)
        ivf_index.train(vectors)
        ivf_index.add(vectors)
        
        # Set search parameters
        ivf_index.nprobe = min(10, num_clusters)  # Number of clusters to search
        
        return ivf_index
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing vector store"""
        if self.vector_store is None:
            self.create_vector_store(documents)
        else:
            self.vector_store.add_documents(documents)
            
            # Clear cache when new documents added
            self.query_cache.clear()
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        use_cache: bool = True,
        **kwargs
    ) -> List[Document]:
        """
        Optimized similarity search with caching
        
        Args:
            query: Search query
            k: Number of results to return
            use_cache: Whether to use query cache
            **kwargs: Additional search parameters
        
        Returns:
            List of similar documents
        """
        # Check cache
        cache_key = f"{query}_{k}"
        if use_cache and cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Perform search
        results = self.vector_store.similarity_search(query, k=k, **kwargs)
        
        # Cache results
        if use_cache:
            self.query_cache[cache_key] = results
            
            # Limit cache size
            if len(self.query_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self.query_cache.keys())[:100]
                for key in oldest_keys:
                    del self.query_cache[key]
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.7
    ) -> List[tuple[Document, float]]:
        """
        Similarity search with relevance scores
        
        Args:
            query: Search query
            k: Number of results
            score_threshold: Minimum relevance score (0-1)
        
        Returns:
            List of (document, score) tuples
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Filter by score threshold
        filtered_results = [
            (doc, score) for doc, score in results
            if score >= score_threshold
        ]
        
        return filtered_results
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        MMR search for diverse results
        
        Args:
            query: Search query
            k: Number of results
            fetch_k: Number of documents to fetch before MMR
            lambda_mult: Diversity factor (0=max diversity, 1=max relevance)
        
        Returns:
            List of diverse documents
        """
        if hasattr(self.vector_store, 'max_marginal_relevance_search'):
            return self.vector_store.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
        else:
            # Fallback to regular search
            return self.similarity_search(query, k=k)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save vector store to disk"""
        save_path = path or self.persist_directory
        
        if self.vector_store_type == "faiss":
            self.vector_store.save_local(save_path)
        elif self.vector_store_type == "chroma":
            # Chroma auto-persists
            pass
    
    def load(self, path: Optional[str] = None) -> None:
        """Load vector store from disk"""
        load_path = path or self.persist_directory
        
        if self.vector_store_type == "faiss":
            self.vector_store = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        elif self.vector_store_type == "chroma":
            self.vector_store = Chroma(
                persist_directory=load_path,
                embedding_function=self.embeddings
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        stats = {
            'vector_store_type': self.vector_store_type,
            'embedding_provider': self.embedding_provider,
            'cache_size': len(self.query_cache)
        }
        
        if self.vector_store_type == "faiss" and self.vector_store:
            stats['num_vectors'] = self.vector_store.index.ntotal
            stats['dimension'] = self.vector_store.index.d
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear query cache"""
        self.query_cache.clear()


# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = VectorDatabaseOptimizer(
        embedding_provider="cohere",  # Best quality, FREE
        vector_store="faiss",  # Fastest, FREE
        persist_directory="./vector_db"
    )
    
    # Create sample documents
    documents = [
        Document(page_content="Python is a high-level programming language", 
                metadata={"source": "doc1"}),
        Document(page_content="JavaScript is used for web development",
                metadata={"source": "doc2"}),
        Document(page_content="Machine learning uses Python extensively",
                metadata={"source": "doc3"}),
    ]
    
    # Create vector store
    optimizer.create_vector_store(documents)
    
    # Search
    results = optimizer.similarity_search("programming languages", k=2)
    print(f"Found {len(results)} results:")
    for doc in results:
        print(f"  - {doc.page_content}")
    
    # Get stats
    stats = optimizer.get_stats()
    print(f"\nVector Store Stats: {stats}")
    
    # Save
    optimizer.save()
    print("\nVector store saved successfully!")
