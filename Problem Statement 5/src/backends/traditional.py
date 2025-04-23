import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any, Tuple

class TraditionalSearchBackend:
    """Traditional search backend using BM25 and TF-IDF"""
    
    def __init__(self, method: str = "bm25"):
        """
        Initialize the traditional search backend
        
        Args:
            method: The search method to use ('bm25' or 'tfidf')
        """
        self.method = method
        self.corpus = None
        self.document_ids = None
        self.tokenized_corpus = None
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index the documents for search
        
        Args:
            documents: List of document dictionaries with 'id', 'title', and 'content' fields
        """
        self.corpus = []
        self.document_ids = []
        
        for doc in documents:
            content = f"{doc['title']} {doc['content']}"
            self.corpus.append(content)
            self.document_ids.append(doc['id'])
        
        # Tokenize documents for BM25
        if self.method == "bm25":
            self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Create TF-IDF matrix for TF-IDF method
        if self.method == "tfidf":
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents using traditional IR methods
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with document information and scores
        """
        if self.method == "bm25":
            return self._bm25_search(query, top_k)
        else:
            return self._tfidf_search(query, top_k)
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25 search implementation"""
        if not self.bm25:
            raise ValueError("Documents not indexed. Call index_documents first.")
            
        # Tokenize query and get BM25 scores
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k document indices
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                results.append({
                    'id': self.document_ids[idx],
                    'score': float(doc_scores[idx]),
                    'method': 'BM25',
                    'rank': len(results) + 1
                })
                
        return results
    
    def _tfidf_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """TF-IDF search implementation"""
        if not self.tfidf_vectorizer:
            raise ValueError("Documents not indexed. Call index_documents first.")
            
        # Transform query to TF-IDF space
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate cosine similarity between query and documents
        cosine_similarities = np.dot(query_vector, self.tfidf_matrix.T).toarray().flatten()
        
        # Get top k document indices
        top_indices = np.argsort(cosine_similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if cosine_similarities[idx] > 0:
                results.append({
                    'id': self.document_ids[idx],
                    'score': float(cosine_similarities[idx]),
                    'method': 'TF-IDF',
                    'rank': len(results) + 1
                })
                
        return results