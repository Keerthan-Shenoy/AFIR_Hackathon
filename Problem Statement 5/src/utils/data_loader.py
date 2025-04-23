import json
import os
from typing import List, Dict, Any

def load_documents(file_path: str) -> List[Dict[str, Any]]:
    """
    Load documents from a JSON file
    
    Args:
        file_path: Path to JSON file with documents
        
    Returns:
        List of document dictionaries
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document file not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
        
    # Ensure each document has required fields
    for doc in documents:
        if 'id' not in doc:
            raise ValueError("Each document must have an 'id' field")
        if 'title' not in doc:
            doc['title'] = f"Document {doc['id']}"
        if 'content' not in doc:
            raise ValueError("Each document must have a 'content' field")
            
    return documents