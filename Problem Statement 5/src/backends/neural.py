import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util

class NeuralSearchBackend:
    """Neural search backend using transformer-based models"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the neural search backend
        
        Args:
            model_name: Name of the transformer model to use
        """
        self.model_name = model_name
        self.document_ids = None
        self.corpus = None
        self.document_embeddings = None
        
        # Load model
        if "sentence-transformers" in model_name:
            self.model = SentenceTransformer(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
    
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
        
        # Create document embeddings
        if isinstance(self.model, SentenceTransformer):
            self.document_embeddings = self.model.encode(self.corpus, convert_to_tensor=True)
        else:
            self.document_embeddings = self._custom_encode(self.corpus)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents using neural embedding similarity
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with document information and scores
        """
        if self.document_embeddings is None:
            raise ValueError("Documents not indexed. Call index_documents first.")
        
        # Encode query
        if isinstance(self.model, SentenceTransformer):
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            # Calculate cosine similarity
            cos_scores = util.cos_sim(query_embedding, self.document_embeddings)[0]
            # Get top results
            top_results = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))
            
            results = []
            for score, idx in zip(top_results[0], top_results[1]):
                results.append({
                    'id': self.document_ids[idx],
                    'score': float(score),
                    'method': 'Neural (SBERT)',
                    'rank': len(results) + 1
                })
        else:
            query_embedding = self._custom_encode([query])[0]
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(self.document_embeddings):
                similarity = torch.cosine_similarity(query_embedding, doc_embedding, dim=0)
                similarities.append((similarity, i))
            
            # Sort by similarity
            similarities.sort(reverse=True)
            
            results = []
            for i, (score, idx) in enumerate(similarities[:top_k]):
                results.append({
                    'id': self.document_ids[idx],
                    'score': float(score),
                    'method': 'Neural (Custom)',
                    'rank': i + 1
                })
                
        return results
    
    def _custom_encode(self, texts: List[str]) -> List[torch.Tensor]:
        """Custom encoding for models that aren't sentence-transformers"""
        embeddings = []
        
        for text in texts:
            # Tokenize and encode
            inputs = self.tokenizer(text, return_tensors="pt", 
                                    max_length=512, padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use mean pooling of the last hidden state
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze())
            
        return embeddings

# ColBERT implementation (a more complex neural IR approach)
class ColBERTSearchBackend(NeuralSearchBackend):
    """ColBERT search backend - a more advanced neural search approach"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__(model_name)
        self.max_tokens = 512
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents using token-level representations"""
        self.corpus = []
        self.document_ids = []
        self.document_token_embeddings = []
        
        for doc in documents:
            content = f"{doc['title']} {doc['content']}"
            self.corpus.append(content)
            self.document_ids.append(doc['id'])
        
        for doc in self.corpus:
            # Get token-level embeddings for each document
            inputs = self.tokenizer(doc, return_tensors="pt", 
                                   max_length=self.max_tokens, padding="max_length", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Store token representations (last hidden state)
            token_embeddings = outputs.last_hidden_state.squeeze()
            self.document_token_embeddings.append(token_embeddings)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using late interaction (ColBERT-style)"""
        # Get query token embeddings
        inputs = self.tokenizer(query, return_tensors="pt", 
                              max_length=32, padding="max_length", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        query_token_embeddings = outputs.last_hidden_state.squeeze()
        
        # Calculate maximum similarity for each query token with each document
        scores = []
        for doc_id, doc_tokens in zip(self.document_ids, self.document_token_embeddings):
            # For each query token, find its maximum similarity with document tokens
            similarity_matrix = torch.matmul(query_token_embeddings, doc_tokens.transpose(0, 1))
            max_similarities = torch.max(similarity_matrix, dim=1)[0]
            # Sum similarities across query tokens
            score = max_similarities.sum().item()
            scores.append((doc_id, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (doc_id, score) in enumerate(scores[:top_k]):
            results.append({
                'id': doc_id,
                'score': score,
                'method': 'Neural (ColBERT)',
                'rank': i + 1
            })
                
        return results