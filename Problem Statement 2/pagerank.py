import networkx as nx
import matplotlib.pyplot as plt
import os

def compute_pagerank(G, alpha=0.85, max_iter=100, tol=1.0e-6):
    N = G.number_of_nodes()
    pr = {node: 1/N for node in G}
    history = [pr.copy()]

    for _ in range(max_iter):
        new_pr = {}
        for node in G:
            rank_sum = sum(pr[nbr] / len(G[nbr]) for nbr in G.predecessors(node))
            new_pr[node] = (1 - alpha) / N + alpha * rank_sum
        history.append(new_pr.copy())

        if all(abs(new_pr[n] - pr[n]) < tol for n in pr):
            break
        pr = new_pr

    return pr, history

def generate_convergence_plot(history):
    plt.figure()
    for node in history[0]:
        plt.plot([h[node] for h in history], label=node)
    plt.xlabel("Iteration")
    plt.ylabel("PageRank Score")
    plt.title("Convergence of PageRank")
    plt.legend()
    path = "static/convergence.png"
    plt.savefig(path)
    plt.close()
    return path
