import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
import pandas as pd

def visualize_comparison(results: Dict[str, Any]) -> None:
    """
    Visualize the comparison between traditional and neural search results
    
    Args:
        results: Results dictionary from search method
    """
    traditional = results['traditional_results']
    neural = results['neural_results']
    query = results['query']
    
    # Extract document IDs and scores
    trad_ids = [result['id'] for result in traditional]
    trad_scores = [result['score'] for result in traditional]
    trad_titles = [result['title'] for result in traditional]
    
    neural_ids = [result['id'] for result in neural]
    neural_scores = [result['score'] for result in neural]
    neural_titles = [result['title'] for result in neural]
    
    # Create normalized scores for better comparison
    if trad_scores:
        trad_scores_norm = [score/max(trad_scores) for score in trad_scores]
    else:
        trad_scores_norm = []
        
    if neural_scores:
        neural_scores_norm = [score/max(neural_scores) for score in neural_scores]
    else:
        neural_scores_norm = []
    
    # Set up plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot traditional results
    if trad_scores:
        trad_y_pos = np.arange(len(trad_titles))
        ax[0].barh(trad_y_pos, trad_scores_norm, align='center')
        ax[0].set_yticks(trad_y_pos)
        ax[0].set_yticklabels(trad_titles)
        ax[0].invert_yaxis()  # labels read top-to-bottom
        ax[0].set_xlabel('Score (normalized)')
        ax[0].set_title('Traditional Search Results')
    else:
        ax[0].set_title('No Traditional Search Results')
    
    # Plot neural results
    if neural_scores:
        neural_y_pos = np.arange(len(neural_titles))
        ax[1].barh(neural_y_pos, neural_scores_norm, align='center', color='orange')
        ax[1].set_yticks(neural_y_pos)
        ax[1].set_yticklabels(neural_titles)
        ax[1].invert_yaxis()  # labels read top-to-bottom
        ax[1].set_xlabel('Score (normalized)')
        ax[1].set_title('Neural Search Results')
    else:
        ax[1].set_title('No Neural Search Results')
    
    plt.tight_layout()
    plt.suptitle(f"Search results for: '{query}'", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_feedback_comparison(feedback_data: pd.DataFrame) -> None:
    """
    Plot feedback comparison between traditional and neural search
    
    Args:
        feedback_data: DataFrame containing feedback data
    """
    if feedback_data.empty:
        print("No feedback data available for visualization")
        return
    
    # Group by method and calculate mean rating
    method_ratings = feedback_data.groupby('method')['rating'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.bar(method_ratings['method'], method_ratings['rating'], color=['blue', 'orange'])
    plt.ylim(0, 5)
    plt.xlabel('Search Method')
    plt.ylabel('Average Rating')
    plt.title('User Feedback: Traditional vs. Neural Search')
    plt.show()
    
    # Plot ratings by rank position
    plt.figure(figsize=(12, 6))
    
    # Group by method and rank
    rank_ratings = feedback_data.groupby(['method', 'rank'])['rating'].mean().reset_index()
    
    # Split by method
    trad_data = rank_ratings[rank_ratings['method'] == 'traditional']
    neural_data = rank_ratings[rank_ratings['method'] == 'neural']
    
    plt.plot(trad_data['rank'], trad_data['rating'], 'o-', label='Traditional')
    plt.plot(neural_data['rank'], neural_data['rating'], 'o-', label='Neural')
    plt.xlabel('Result Rank')
    plt.ylabel('Average Rating')
    plt.title('Rating by Result Position')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()