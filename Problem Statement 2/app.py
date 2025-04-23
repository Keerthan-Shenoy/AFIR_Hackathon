from flask import Flask, render_template, request, redirect, url_for
import networkx as nx
from pagerank import compute_pagerank, generate_convergence_plot

app = Flask(__name__)
graph = nx.DiGraph()

@app.route("/", methods=["GET", "POST"])
def index():
    global graph
    if request.method == "POST":
        edge_list = request.form["edges"].strip().split("\n")
        graph = nx.DiGraph()
        for line in edge_list:
            parts = line.strip().split()
            if len(parts) == 2:
                graph.add_edge(parts[0], parts[1])
        return redirect(url_for("visualize"))

    return render_template("index.html")

@app.route("/visualize")
def visualize():
    ranks, history = compute_pagerank(graph)
    plot_path = generate_convergence_plot(history)
    return render_template("visualize.html", ranks=ranks, plot_path=plot_path, graph=graph.edges)

@app.route("/update", methods=["POST"])
def update():
    action = request.form["action"]
    src = request.form["source"]
    dst = request.form["target"]
    if action == "add":
        graph.add_edge(src, dst)
    elif action == "remove":
        graph.remove_edge(src, dst)
    return redirect(url_for("visualize"))

if __name__ == "__main__":
    app.run(debug=True, port=5001   )
