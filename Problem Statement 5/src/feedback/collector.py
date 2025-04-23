import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import os

class FeedbackCollector:
    """Collects and manages user feedback on search results"""
    
    def __init__(self, feedback_file: str = "feedback_data.csv"):
        """
        Initialize feedback collector
        
        Args:
            feedback_file: Path to store feedback data
        """
        self.feedback_file = feedback_file
        self.feedback_data = []
        
        # Load existing feedback if file exists
        if os.path.exists(feedback_file):
            self.load_feedback()
    
    def add_feedback(self, query: str, doc_id: int, method: str, 
                    rank: int, rating: int) -> None:
        """
        Add a new feedback entry
        
        Args:
            query: The search query
            doc_id: Document ID
            method: Search method ('traditional' or 'neural')
            rank: Result rank position
            rating: User rating (1-5)
        """
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'doc_id': doc_id,
            'method': method,
            'rank': rank,
            'rating': rating
        }
        
        self.feedback_data.append(feedback)
        self.save_feedback()
    
    def load_feedback(self) -> pd.DataFrame:
        """
        Load feedback data from file
        
        Returns:
            DataFrame with feedback data
        """
        try:
            df = pd.read_csv(self.feedback_file)
            self.feedback_data = df.to_dict('records')
            return df
        except Exception as e:
            print(f"Error loading feedback data: {e}")
            return pd.DataFrame()
    
    def save_feedback(self) -> None:
        """Save feedback data to file"""
        df = pd.DataFrame(self.feedback_data)
        try:
            df.to_csv(self.feedback_file, index=False)
        except Exception as e:
            print(f"Error saving feedback data: {e}")
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of collected feedback
        
        Returns:
            Dictionary with feedback summary statistics
        """
        if not self.feedback_data:
            return {'message': 'No feedback data available'}
            
        df = pd.DataFrame(self.feedback_data)
        
        summary = {
            'total_feedback': len(df),
            'unique_queries': df['query'].nunique(),
            'traditional': {
                'count': len(df[df['method'] == 'traditional']),
                'avg_rating': df[df['method'] == 'traditional']['rating'].mean()
            },
            'neural': {
                'count': len(df[df['method'] == 'neural']),
                'avg_rating': df[df['method'] == 'neural']['rating'].mean()
            }
        }
        
        return summary
    
    def get_feedback_by_query(self, query: str) -> pd.DataFrame:
        """
        Get feedback for a specific query
        
        Args:
            query: The search query to filter by
            
        Returns:
            DataFrame with filtered feedback data
        """
        df = pd.DataFrame(self.feedback_data)
        if df.empty:
            return pd.DataFrame()
            
        return df[df['query'] == query]
    
    def improve_rankings(self, query: str, method: str) -> List[int]:
        """
        Use feedback data to improve document rankings for a query
        
        Args:
            query: The search query
            method: Search method ('traditional' or 'neural')
            
        Returns:
            List of document IDs in improved ranking order
        """
        feedback = self.get_feedback_by_query(query)
        if feedback.empty:
            return []
            
        # Filter by method
        method_feedback = feedback[feedback['method'] == method]
        if method_feedback.empty:
            return []
        
        # Sort by rating (descending)
        sorted_docs = method_feedback.sort_values('rating', ascending=False)
        
        return sorted_docs['doc_id'].tolist()