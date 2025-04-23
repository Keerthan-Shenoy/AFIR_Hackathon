import json
import os
from datasets import load_dataset

# Path where to save your dataset
output_path = r"d:\AFIR Hackathon\6\data\sample_documents.json"

# Create simple sample documents
docs = [
    {
        'id': 1,
        'title': 'PageRank Algorithm',
        'content': 'PageRank is an algorithm used by Google Search to rank web pages in their search engine results.'
    },
    {
        'id': 2,
        'title': 'BERT NLP Model',
        'content': 'BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for NLP.'
    },
    # Add more sample documents as needed
]

# Save as JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(docs, f, indent=2)

print(f"Created dataset with {len(docs)} documents at {output_path}")