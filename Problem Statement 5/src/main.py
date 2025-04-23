import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from backends.traditional import TraditionalSearchBackend
from backends.neural import NeuralSearchBackend, ColBERTSearchBackend
from utils.data_loader import load_documents
from utils.visualization import visualize_comparison
from feedback.collector import FeedbackCollector

class ComparativeSearchSystem:
    """Main system that compares traditional and neural search results"""
    
    def __init__(self, 
                 traditional_method: str = "bm25",
                 neural_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the comparative search system
        
        Args:
            traditional_method: Method for traditional search ('bm25' or 'tfidf')
            neural_model: Transformer model name for neural search
        """
        self.traditional_backend = TraditionalSearchBackend(method=traditional_method)
        self.neural_backend = NeuralSearchBackend(model_name=neural_model)
        self.documents = None
        self.document_map = {}
        self.feedback_collector = FeedbackCollector()
        
    def load_and_index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Load and index documents for both backends
        
        Args:
            documents: List of document dictionaries
        """
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        
        print("Indexing documents for traditional search...")
        self.traditional_backend.index_documents(documents)
        
        print("Indexing documents for neural search...")
        self.neural_backend.index_documents(documents)
        
        print(f"Indexed {len(documents)} documents")
        
    def search(self, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search using both backends and return results
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            Dictionary with results from both backends
        """
        print(f"Searching for: '{query}'")
        
        traditional_results = self.traditional_backend.search(query, top_k)
        neural_results = self.neural_backend.search(query, top_k)
        
        # Add document content to results
        for result in traditional_results + neural_results:
            doc_id = result['id']
            if doc_id in self.document_map:
                result['title'] = self.document_map[doc_id]['title']
                result['content'] = self.document_map[doc_id]['content']
        
        return {
            'query': query,
            'traditional_results': traditional_results,
            'neural_results': neural_results
        }
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """
        Display search results in a user-friendly format
        
        Args:
            results: Results dictionary from search method
        """
        query = results['query']
        traditional = results['traditional_results']
        neural = results['neural_results']
        
        print("\n" + "="*80)
        print(f"Search Results for: '{query}'")
        print("="*80)
        
        print("\nTraditional Search Results:")
        for i, result in enumerate(traditional):
            print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
            print(f"   {result['content'][:100]}...")
        
        print("\nNeural Search Results:")
        for i, result in enumerate(neural):
            print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
            print(f"   {result['content'][:100]}...")
    
    def collect_feedback(self, results: Dict[str, Any]) -> None:
        """
        Collect user feedback on search results
        
        Args:
            results: Results dictionary from search method
        """
        query = results['query']
        print("\nPlease rate the relevance of each result (1-5, or 0 to skip):")
        
        # Collect feedback for traditional results
        print("\nTraditional Search Results:")
        for i, result in enumerate(results['traditional_results']):
            rating = input(f"{i+1}. {result['title']} - Rating (0-5): ")
            try:
                rating = int(rating)
                if 1 <= rating <= 5:
                    self.feedback_collector.add_feedback(
                        query=query,
                        doc_id=result['id'],
                        method="traditional",
                        rank=result['rank'],
                        rating=rating
                    )
            except ValueError:
                pass
        
        # Collect feedback for neural results
        print("\nNeural Search Results:")
        for i, result in enumerate(results['neural_results']):
            rating = input(f"{i+1}. {result['title']} - Rating (0-5): ")
            try:
                rating = int(rating)
                if 1 <= rating <= 5:
                    self.feedback_collector.add_feedback(
                        query=query,
                        doc_id=result['id'],
                        method="neural",
                        rank=result['rank'],
                        rating=rating
                    )
            except ValueError:
                pass

def main():
    parser = argparse.ArgumentParser(description="Comparative Search System")
    parser.add_argument("--data", type=str, default="data/sample_documents.json", 
                        help="Path to document corpus JSON file")
    parser.add_argument("--traditional", type=str, default="bm25", 
                        choices=["bm25", "tfidf"], help="Traditional search method")
    parser.add_argument("--neural", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Neural model name")
    args = parser.parse_args()
    
    # Create search system
    search_system = ComparativeSearchSystem(
        traditional_method=args.traditional,
        neural_model=args.neural
    )
    
    # Load documents
    try:
        documents = load_documents(args.data)
    except FileNotFoundError:
        # Create sample documents if file doesn't exist
        documents = [
            # PageRank Algorithm (25 documents)
            {
                'id': 1,
                'title': 'PageRank Algorithm Introduction',
                'content': 'PageRank is an algorithm used by Google Search to rank web pages in their search engine results. It works by counting the number and quality of links to a page.'
            },
            {
                'id': 2,
                'title': 'PageRank History',
                'content': 'PageRank was developed at Stanford University by Larry Page and Sergey Brin in 1996 as part of a research project about a new kind of search engine.'
            },
            {
                'id': 3,
                'title': 'Mathematics Behind PageRank',
                'content': 'PageRank uses the random surfer model, representing the behavior of a web surfer who randomly clicks on links. It can be calculated using a Markov chain model.'
            },
            {
                'id': 4,
                'title': 'PageRank and Link Analysis',
                'content': 'PageRank is a type of link analysis algorithm that assigns a numerical weighting to each element of a hyperlinked set of documents, such as the World Wide Web.'
            },
            {
                'id': 5,
                'title': 'PageRank Damping Factor',
                'content': 'The damping factor in PageRank represents the probability that a random surfer continues clicking links rather than starting a new random page.'
            },
            {
                'id': 6,
                'title': 'PageRank Implementation',
                'content': 'Implementing PageRank involves creating an adjacency matrix of web pages and iteratively calculating the rank vector until convergence.'
            },
            {
                'id': 7,
                'title': 'PageRank vs HITS Algorithm',
                'content': 'While PageRank assigns a single score to each page, HITS (Hyperlink-Induced Topic Search) assigns two scores: authority and hub values.'
            },
            {
                'id': 8,
                'title': 'PageRank in Modern SEO',
                'content': 'Though Google has evolved beyond pure PageRank, link authority concepts from PageRank remain fundamental to how search engines evaluate website quality.'
            },
            {
                'id': 9,
                'title': 'Personalized PageRank',
                'content': 'Personalized PageRank modifies the original algorithm to be biased toward certain nodes, creating topic-specific or user-specific rankings.'
            },
            {
                'id': 10,
                'title': 'PageRank Convergence',
                'content': 'The PageRank computation converges because it forms an irreducible and aperiodic Markov chain, guaranteeing a unique stationary distribution.'
            },
            {
                'id': 11,
                'title': 'PageRank Applications',
                'content': 'Beyond web search, PageRank has been applied to social network analysis, bibliometrics, recommendation systems, and biology networks.'
            },
            {
                'id': 12,
                'title': 'PageRank Limitations',
                'content': 'PageRank can be manipulated through link farms, lacks consideration of content relevance, and doesnt account for user behavior or content freshness.'
            },
            {
                'id': 13,
                'title': 'PageRank in Python',
                'content': 'NetworkX library in Python provides built-in functions to calculate PageRank on graph structures with minimal coding effort.'
            },
            {
                'id': 14,
                'title': 'PageRank Patent',
                'content': 'The PageRank algorithm was patented by Stanford University in 1998 and exclusively licensed to Google until the patent expired in 2019.'
            },
            {
                'id': 15,
                'title': 'PageRank and Random Walks',
                'content': 'PageRank can be interpreted as measuring the likelihood of landing on a particular page during a random walk through the web graph.'
            },
            {
                'id': 16,
                'title': 'TrustRank vs PageRank',
                'content': 'TrustRank is a variant of PageRank that aims to combat web spam by prioritizing links from trusted seed pages over general link popularity.'
            },
            {
                'id': 17,
                'title': 'PageRank Time Complexity',
                'content': 'The time complexity of PageRank computation depends on the sparsity of the web graph and the number of iterations required for convergence.'
            },
            {
                'id': 18,
                'title': 'PageRank for Citation Analysis',
                'content': 'PageRank principles have been applied to academic citation networks to identify influential papers and researchers beyond simple citation counts.'
            },
            {
                'id': 19,
                'title': 'Incremental PageRank',
                'content': 'Incremental PageRank algorithms update rankings efficiently when the underlying web graph changes, avoiding full recomputation.'
            },
            {
                'id': 20,
                'title': 'PageRank Visualization',
                'content': 'Visualizing PageRank often uses node size or color intensity in graph visualizations to represent the relative importance of web pages.'
            },
            {
                'id': 21,
                'title': 'PageRank and Web Crawling',
                'content': 'Search engines use PageRank to prioritize which pages to crawl, focusing resources on more important pages in the web graph.'
            },
            {
                'id': 22,
                'title': 'PageRank Computation Methods',
                'content': 'PageRank can be computed using power iteration, Monte Carlo methods, or algebraic approaches solving the eigenvector problem directly.'
            },
            {
                'id': 23,
                'title': 'Topic-Sensitive PageRank',
                'content': 'Topic-Sensitive PageRank creates multiple rank vectors biased toward different topics to provide context-aware search results.'
            },
            {
                'id': 24,
                'title': 'PageRank in Distributed Systems',
                'content': 'Distributed implementations of PageRank use frameworks like MapReduce to handle massive web graphs across multiple machines.'
            },
            {
                'id': 25,
                'title': 'Future of PageRank',
                'content': 'Modern search engines combine PageRank concepts with machine learning, user behavior signals, and content analysis for more nuanced rankings.'
            },

            # BERT NLP Model (25 documents)
            {
                'id': 26,
                'title': 'BERT NLP Model Introduction',
                'content': 'BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for natural language processing developed by Google.'
            },
            {
                'id': 27,
                'title': 'BERT Architecture',
                'content': 'BERT uses a transformer architecture with self-attention mechanisms that learn contextual representations of words by looking at both left and right contexts.'
            },
            {
                'id': 28,
                'title': 'BERT Pre-training',
                'content': 'BERT is pre-trained on a large corpus using masked language modeling (predicting masked words) and next sentence prediction tasks.'
            },
            {
                'id': 29,
                'title': 'BERT Fine-tuning',
                'content': 'After pre-training, BERT can be fine-tuned on specific tasks like sentiment analysis, question answering, and document classification with minimal additional training.'
            },
            {
                'id': 30,
                'title': 'BERT vs GPT Models',
                'content': 'While BERT uses bidirectional context (looking at both sides of a word), GPT models are unidirectional, generating text by predicting the next token based on previous context.'
            },
            {
                'id': 31,
                'title': 'BERT Tokenization',
                'content': 'BERT uses WordPiece tokenization, breaking words into subword units to handle out-of-vocabulary words and improve representation of rare words.'
            },
            {
                'id': 32,
                'title': 'BERT for Question Answering',
                'content': 'BERT achieves state-of-the-art performance on question answering tasks by encoding question and context together and predicting answer spans.'
            },
            {
                'id': 33,
                'title': 'BERT for Sentiment Analysis',
                'content': 'Fine-tuned BERT models can accurately classify the sentiment of text by understanding the contextual nuances and linguistic patterns.'
            },
            {
                'id': 34,
                'title': 'BERT Variants',
                'content': 'Various BERT variants like RoBERTa, ALBERT, and DistilBERT have been developed to improve performance, efficiency, or handle specific languages.'
            },
            {
                'id': 35,
                'title': 'BERT Attention Mechanism',
                'content': 'BERTs self-attention mechanism allows it to weigh the importance of each word in relation to all other words in a sentence, capturing complex dependencies.'
            },
            {
                'id': 36,
                'title': 'Multilingual BERT',
                'content': 'mBERT is trained on text from 104 languages and can perform cross-lingual tasks, transferring knowledge between languages with shared structures.'
            },
            {
                'id': 37,
                'title': 'BERT Limitations',
                'content': 'BERT has limitations including input length constraints, computational demands, difficulty with specialized domains, and challenges understanding common sense.'
            },
            {
                'id': 38,
                'title': 'BERT and Transfer Learning',
                'content': 'BERT exemplifies transfer learning in NLP, where knowledge from general language understanding transfers to specific downstream tasks with minimal adaptation.'
            },
            {
                'id': 39,
                'title': 'BERT Embeddings',
                'content': 'BERT produces contextual word embeddings where the same word can have different vector representations depending on its surrounding context.'
            },
            {
                'id': 40,
                'title': 'BERT for Text Classification',
                'content': 'BERT excels at text classification tasks by encoding the entire text and using the [CLS] token representation for making classification decisions.'
            },
            {
                'id': 41,
                'title': 'BERT Training Data',
                'content': 'The original BERT was trained on BookCorpus (800M words) and English Wikipedia (2,500M words), helping it learn diverse language patterns.'
            },
            {
                'id': 42,
                'title': 'BERT and Contextual Meaning',
                'content': 'BERT can disambiguate words with multiple meanings based on context, correctly interpreting "bank" as a financial institution or river edge.'
            },
            {
                'id': 43,
                'title': 'BERT Model Sizes',
                'content': 'BERT comes in different sizes, with BERT-base (110M parameters) and BERT-large (340M parameters) offering trade-offs between accuracy and efficiency.'
            },
            {
                'id': 44,
                'title': 'BERT for Named Entity Recognition',
                'content': 'BERT achieves strong performance on NER by treating it as a token classification task, labeling each token with its entity type.'
            },
            {
                'id': 45,
                'title': 'BERT Position Embeddings',
                'content': 'BERT uses learned position embeddings that are added to token embeddings to maintain awareness of word order in the transformer architecture.'
            },
            {
                'id': 46,
                'title': 'BERT in Production Systems',
                'content': 'Deploying BERT in production systems often requires optimization techniques like distillation, pruning, or quantization to meet latency requirements.'
            },
            {
                'id': 47,
                'title': 'BERT Evaluation Metrics',
                'content': 'BERTs performance is evaluated using task-specific metrics like accuracy, F1 score, BLEU, or ROUGE depending on the application.'
            },
            {
                'id': 48,
                'title': 'BERT for Document Retrieval',
                'content': 'BERT can power semantic search systems by encoding queries and documents in the same space for meaning-based retrieval rather than lexical matching.'
            },
            {
                'id': 49,
                'title': 'BERT and Language Understanding',
                'content': 'BERT dramatically improved machine language understanding by capturing contextual relationships and linguistic structures from massive text corpora.'
            },
            {
                'id': 50,
                'title': 'BERT vs Traditional Word Embeddings',
                'content': 'Unlike static word embeddings like Word2Vec, BERT generates dynamic contextual representations where a word embedding changes based on its context.'
            },

            # Python Programming (25 documents)
            {
                'id': 51,
                'title': 'Python Programming Introduction',
                'content': 'Python is an interpreted, high-level and general-purpose programming language known for its readability and simple syntax.'
            },
            {
                'id': 52,
                'title': 'Python Data Types',
                'content': 'Python includes built-in data types like integers, floats, strings, lists, tuples, dictionaries, and sets, each with specific use cases and methods.'
            },
            {
                'id': 53,
                'title': 'Python Functions',
                'content': 'Functions in Python are defined using the def keyword and can accept parameters, return values, and use default arguments or variable-length argument lists.'
            },
            {
                'id': 54,
                'title': 'Python Classes and Objects',
                'content': 'Python supports object-oriented programming with classes, inheritance, polymorphism, and encapsulation to organize code and data.'
            },
            {
                'id': 55,
                'title': 'Python Libraries and Modules',
                'content': 'Python rich ecosystem includes libraries like NumPy, Pandas, Matplotlib, TensorFlow, and Django for scientific computing, data analysis, and web development.'
            },
            {
                'id': 56,
                'title': 'Python File Handling',
                'content': 'Python can read from and write to files using built-in functions like open(), read(), write(), and the with statement for automatic resource management.'
            },
            {
                'id': 57,
                'title': 'Python List Comprehensions',
                'content': 'List comprehensions provide a concise way to create lists based on existing lists, combining for loops and conditional logic in a single expression.'
            },
            {
                'id': 58,
                'title': 'Python Exception Handling',
                'content': 'Try-except blocks in Python catch and handle errors, preventing program crashes and allowing for graceful error recovery.'
            },
            {
                'id': 59,
                'title': 'Python Virtual Environments',
                'content': 'Virtual environments in Python isolate project dependencies, preventing conflicts between packages needed for different projects.'
            },
            {
                'id': 60,
                'title': 'Python Decorators',
                'content': 'Decorators modify function behavior without changing their definition, useful for logging, timing, authentication, and other cross-cutting concerns.'
            },
            {
                'id': 61,
                'title': 'Python Iterators and Generators',
                'content': 'Iterators and generators in Python enable efficient sequence traversal and lazy evaluation, saving memory when working with large datasets.'
            },
            {
                'id': 62,
                'title': 'Python Lambda Functions',
                'content': 'Lambda functions are anonymous single-line functions that can be used wherever function objects are required, especially in functional programming patterns.'
            },
            {
                'id': 63,
                'title': 'Python Regular Expressions',
                'content': 'The re module in Python provides pattern matching capabilities for text processing, validation, and advanced search and replace operations.'
            },
            {
                'id': 64,
                'title': 'Python Threading and Multiprocessing',
                'content': 'Python offers threading and multiprocessing modules for concurrent execution, with multiprocessing bypassing the Global Interpreter Lock for true parallelism.'
            },
            {
                'id': 65,
                'title': 'Python Data Visualization',
                'content': 'Libraries like Matplotlib, Seaborn, and Plotly enable Python users to create static, animated, and interactive visualizations from data.'
            },
            {
                'id': 66,
                'title': 'Python Web Scraping',
                'content': 'Python packages like Beautiful Soup and Scrapy facilitate extracting data from websites by parsing HTML and navigating DOM structures.'
            },
            {
                'id': 67,
                'title': 'Python for Machine Learning',
                'content': 'Python is the leading language for machine learning with libraries like Scikit-learn, TensorFlow, and PyTorch providing tools for model training and evaluation.'
            },
            {
                'id': 68,
                'title': 'Python Package Management',
                'content': 'Pip and conda are package managers for Python that install, update, and manage external libraries from repositories like PyPI.'
            },
            {
                'id': 69,
                'title': 'Python Context Managers',
                'content': 'Context managers using the with statement ensure proper acquisition and release of resources like files, locks, and database connections.'
            },
            {
                'id': 70,
                'title': 'Python Testing Frameworks',
                'content': 'Pytest, unittest, and nose provide testing capabilities for Python code, supporting test discovery, fixtures, and assertions.'
            },
            {
                'id': 71,
                'title': 'Python Database Access',
                'content': 'Python connects to databases using libraries like SQLAlchemy, psycopg2, and pymongo for relational and NoSQL data storage solutions.'
            },
            {
                'id': 72,
                'title': 'Python Functional Programming',
                'content': 'Python supports functional programming with first-class functions, map/filter/reduce operations, and immutable data structures.'
            },
            {
                'id': 73,
                'title': 'Python Web Frameworks',
                'content': 'Flask, Django, and FastAPI are popular Python web frameworks for building everything from simple APIs to complex web applications.'
            },
            {
                'id': 74,
                'title': 'Python Type Hints',
                'content': 'Type hints introduced in Python 3.5+ improve code documentation and enable static type checking with tools like mypy.'
            },
            {
                'id': 75,
                'title': 'Python Asynchronous Programming',
                'content': 'Async and await keywords in Python enable non-blocking I/O operations for improved performance in I/O-bound applications.'
            },

            # Question Answering Systems (25 documents)
            {
                'id': 76,
                'title': 'Question Answering Systems Introduction',
                'content': 'Question answering systems use NLP techniques to automatically answer questions posed by humans in natural language, bridging the gap between human queries and machine understanding.'
            },
            {
                'id': 77,
                'title': 'Open-Domain Question Answering',
                'content': 'Open-domain QA systems answer questions without domain restrictions, often using large knowledge bases or the web as information sources.'
            },
            {
                'id': 78,
                'title': 'Extractive Question Answering',
                'content': 'Extractive QA locates and extracts spans of text from a document that directly answer a question, commonly used in reading comprehension tasks.'
            },
            {
                'id': 79,
                'title': 'Generative Question Answering',
                'content': 'Generative QA systems formulate original answers by synthesizing information rather than extracting verbatim text from source documents.'
            },
            {
                'id': 80,
                'title': 'Knowledge-Based Question Answering',
                'content': 'Knowledge-based QA systems query structured knowledge bases like Wikidata or domain-specific ontologies to retrieve facts and answer questions.'
            },
            {
                'id': 81,
                'title': 'Neural Question Answering Models',
                'content': 'Modern QA systems use neural networks like BERT, T5, or GPT to understand questions and generate or extract answers with high accuracy.'
            },
            {
                'id': 82,
                'title': 'Question Answering Datasets',
                'content': 'Datasets like SQuAD, Natural Questions, and HotpotQA provide training and evaluation benchmarks for question answering systems.'
            },
            {
                'id': 83,
                'title': 'Multi-hop Question Answering',
                'content': 'Multi-hop QA requires reasoning across multiple documents or facts to arrive at an answer, testing systems ability to connect information.'
            },
            {
                'id': 84,
                'title': 'Visual Question Answering',
                'content': 'Visual QA systems answer questions about images, combining computer vision and natural language processing techniques.'
            },
            {
                'id': 85,
                'title': 'Conversational Question Answering',
                'content': 'Conversational QA handles follow-up questions and maintains context across a dialogue, providing a more natural interaction experience.'
            },
            {
                'id': 86,
                'title': 'Question Answering Evaluation Metrics',
                'content': 'QA systems are evaluated using metrics like Exact Match (EM), F1 score, BLEU, ROUGE, and human judgment of answer quality and relevance.'
            },
            {
                'id': 87,
                'title': 'Factoid vs. Non-Factoid Questions',
                'content': 'Factoid questions seek specific facts, while non-factoid questions require explanations, opinions, or complex information synthesis.'
            },
            {
                'id': 88,
                'title': 'Question Answering System Architecture',
                'content': 'QA architectures typically include question analysis, information retrieval, answer extraction or generation, and ranking components.'
            },
            {
                'id': 89,
                'title': 'Question Answering in Search Engines',
                'content': 'Modern search engines incorporate QA capabilities to provide direct answers in featured snippets rather than just returning document links.'
            },
            {
                'id': 90,
                'title': 'Reinforcement Learning for Question Answering',
                'content': 'RL techniques can optimize QA systems by learning from user feedback and interaction patterns rather than static training data.'
            },
            {
                'id': 91,
                'title': 'Question Answering in Virtual Assistants',
                'content': 'Virtual assistants like Siri, Alexa, and Google Assistant use QA systems to respond to user queries and provide information on demand.'
            },
            {
                'id': 92,
                'title': 'Domain-Specific Question Answering',
                'content': 'Domain-specific QA systems focus on narrow fields like medicine, law, or customer support with specialized knowledge and vocabulary.'
            },
            {
                'id': 93,
                'title': 'Question Answering for Reading Comprehension',
                'content': 'Reading comprehension QA tests machine understanding of text by requiring answers from specific passages, similar to human reading tests.'
            },
            {
                'id': 94,
                'title': 'Table-based Question Answering',
                'content': 'Table QA interprets structured data in tables to answer queries requiring numerical reasoning, comparisons, or data aggregation.'
            },
            {
                'id': 95,
                'title': 'Multilingual Question Answering',
                'content': 'Multilingual QA systems can understand questions and provide answers across multiple languages, either through translation or direct multilingual modeling.'
            },
            {
                'id': 96,
                'title': 'Real-time Question Answering',
                'content': 'Real-time QA systems must balance speed and accuracy, often using retrieval-augmented generation to quickly access relevant information.'
            },
            {
                'id': 97,
                'title': 'Explainable Question Answering',
                'content': 'Explainable QA not only provides answers but also explains the reasoning process and cites sources to build user trust and verify accuracy.'
            },
            {
                'id': 98,
                'title': 'Question Answering System Challenges',
                'content': 'QA systems face challenges like ambiguity resolution, handling complex questions, maintaining factual accuracy, and countering hallucinations.'
            },
            {
                'id': 99,
                'title': 'Question Paraphrasing in QA',
                'content': 'QA systems must recognize different phrasings of the same question, mapping varied linguistic forms to the same underlying information need.'
            },
            {
                'id': 100,
                'title': 'Future of Question Answering Systems',
                'content': 'Future QA systems will likely feature multimodal capabilities, more robust reasoning, better factuality, and deeper integration with knowledge bases and the web.'
            }
        ]
    
    # Index documents
    search_system.load_and_index_documents(documents)
    
    # Interactive search
    print("\nComparative Search System")
    print("Enter 'exit' to quit")
    
    while True:
        query = input("\nEnter search query: ")
        if query.lower() == 'exit':
            break
            
        # Perform search
        results = search_system.search(query)
        
        # Display results
        search_system.display_results(results)
        
        # Visualize comparison
        visualize_comparison(results)
        
        # Collect feedback
        if input("\nDo you want to provide feedback? (y/n): ").lower() == 'y':
            search_system.collect_feedback(results)

if __name__ == "__main__":
    main()