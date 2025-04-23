import pandas as pd
import sqlite3
from typing import List, Dict
import re
from rank_bm25 import BM25Okapi

class KnowledgeBase:
    def __init__(self, db_path=":memory:"):
        self.db_path = db_path
        self.conn = None
        self.tokenized_corpus = None
        self.bm25 = None
        self.initialize_db()
        self.populate_sample_data()
        
    def initialize_db(self):
        """Initialize the database connection and create tables if needed"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            content TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            name TEXT,
            type TEXT,
            article_id INTEGER,
            FOREIGN KEY (article_id) REFERENCES articles(id)
        )
        ''')
        
        self.conn.commit()

    def populate_sample_data(self):
        """Populate the database with sample data"""
        cursor = self.conn.cursor()
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM articles")
        if cursor.fetchone()[0] > 0:
            return
            
        # Add sample articles
        sample_articles = [
            (1, "PageRank Algorithm", "PageRank is an algorithm used by Google Search to rank web pages in their search engine results."),
            (2, "BERT NLP Model", "BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for NLP."),
            (3, "Python Programming", "Python is an interpreted, high-level and general-purpose programming language."),
            (4, "Question Answering Systems", "Question answering systems use NLP techniques to automatically answer questions posed by humans in natural language.")
        ]
        
        cursor.executemany("INSERT INTO articles (id, title, content) VALUES (?, ?, ?)", sample_articles)
        
        # Add sample entities
        sample_entities = [
            (1, "PageRank", "ALGORITHM", 1),
            (2, "Google", "ORGANIZATION", 1),
            (3, "BERT", "ALGORITHM", 2),
            (4, "NLP", "FIELD", 2),
            (5, "Python", "LANGUAGE", 3),
            (6, "Question Answering", "SYSTEM", 4)
        ]
        
        cursor.executemany("INSERT INTO entities (id, name, type, article_id) VALUES (?, ?, ?, ?)", sample_entities)
        
        self.conn.commit()
        print("Sample data has been added to the in-memory database")
        
    def load_corpus_for_retrieval(self):
        """Load and prepare corpus for BM25 retrieval"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title, content FROM articles")
        articles = cursor.fetchall()
        
        corpus = []
        self.article_map = {}
        
        for article_id, title, content in articles:
            document = f"{title}. {content}"
            corpus.append(document)
            self.article_map[len(corpus) - 1] = article_id
        
        # Tokenize and prepare for BM25
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents based on query using BM25"""
        if not self.bm25:
            self.load_corpus_for_retrieval()
            
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k document indices
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            article_id = self.article_map[idx]
            cursor = self.conn.cursor()
            cursor.execute("SELECT title, content FROM articles WHERE id = ?", (article_id,))
            title, content = cursor.fetchone()
            
            results.append({
                "id": article_id,
                "title": title, 
                "content": content,
                "score": doc_scores[idx]
            })
            
        return results