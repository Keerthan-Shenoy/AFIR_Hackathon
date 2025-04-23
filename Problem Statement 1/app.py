from flask import Flask, render_template, request
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict, Counter
import re
import string
import math

app = Flask(__name__)


# ==== Core Boolean Search Engine Classes ====

class DocumentProcessor:
    def __init__(self):
        self.stopwords = set([
            "a", "an", "the", "and", "or", "not", "in", "on", "at", "to", "for", "with",
            "by", "from", "of", "as", "is", "are", "was", "were", "be", "been", "it", "this", "that"
        ])

    def process_text(self, text):
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        tokens = text.split()
        return [token for token in tokens if token not in self.stopwords]


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(dict)  # term -> {doc_id: term_freq}
        self.doc_lengths = {}  # doc_id -> doc length
        self.total_docs = 0
    
    def add_document(self, doc_id, title_terms, body_terms):
        # Count term frequencies
        term_freq = Counter(title_terms + body_terms)
        self.doc_lengths[doc_id] = len(title_terms + body_terms)
        
        for term, freq in term_freq.items():
            self.index[term][doc_id] = freq
        self.total_docs += 1
    
    def get_tfidf_score(self, term, doc_id):
        if term not in self.index or doc_id not in self.index[term]:
            return 0.0
            
        tf = self.index[term][doc_id]
        idf = math.log(self.total_docs / len(self.index[term]))
        return tf * idf

    def search(self, query_terms, operator='AND'):
        result_sets = []
        for term in query_terms:
            if term.startswith('!'):
                actual_term = term[1:]
                result_sets.append(('NOT', set(self.index.get(actual_term, {}).keys())))
            else:
                result_sets.append(('TERM', set(self.index.get(term, {}).keys())))

        all_doc_ids = set(self.doc_lengths.keys())

        if not result_sets:
            return set()

        result = None
        for op, doc_set in result_sets:
            if op == 'TERM':
                result = doc_set if result is None else (
                    result & doc_set if operator == 'AND' else result | doc_set)
            elif op == 'NOT':
                result = all_doc_ids - doc_set if result is None else result - doc_set

        return result or set()

    def search_phrase(self, phrase_terms):
        if not phrase_terms:
            return set()
        
        # Get docs containing first term
        result = set(self.index[phrase_terms[0]].keys())
        
        # Check each doc for exact phrase
        matched_docs = set()
        for doc_id in result:
            doc = self.documents[doc_id]
            doc_text = doc['title'] + ' ' + doc['body']
            if ' '.join(phrase_terms) in doc_text.lower():
                matched_docs.add(doc_id)
                
        return matched_docs

    def rank_results(self, query_terms, doc_ids):
        query_vec = Counter(query_terms)
        scores = {}
        
        for doc_id in doc_ids:
            score = 0.0
            doc_vec = {}
            
            for term in query_terms:
                q_tfidf = query_vec[term] * math.log(self.total_docs / len(self.index.get(term, {})))
                d_tfidf = self.get_tfidf_score(term, doc_id)
                score += q_tfidf * d_tfidf
                
            # Normalize by doc length
            if self.doc_lengths[doc_id] > 0:
                score /= math.sqrt(self.doc_lengths[doc_id])
                
            scores[doc_id] = score
            
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class ZoneIndex:
    def __init__(self):
        self.title_index = defaultdict(set)
        self.body_index = defaultdict(set)
    
    def add_document(self, doc_id, title_terms, body_terms):
        for term in title_terms:
            self.title_index[term].add(doc_id)
        for term in body_terms:
            self.body_index[term].add(doc_id)
            
    def search(self, term, zone='all'):
        if zone == 'title':
            return self.title_index.get(term, set())
        elif zone == 'body':
            return self.body_index.get(term, set())
        else:
            return self.title_index.get(term, set()) | self.body_index.get(term, set())


class SearchEngine:
    def __init__(self, documents):
        self.processor = DocumentProcessor()
        self.inverted_index = InvertedIndex()
        self.zone_index = ZoneIndex()
        self.documents = {}
        self.raw_documents = documents

    def build_index(self):
        for doc_id, content in enumerate(self.raw_documents):
            parts = content.split('\n', 1)
            title = parts[0] if len(parts) > 0 else ''
            body = parts[1] if len(parts) > 1 else ''
            title_terms = self.processor.process_text(title)
            body_terms = self.processor.process_text(body)
            self.inverted_index.add_document(doc_id, title_terms, body_terms)
            self.zone_index.add_document(doc_id, title_terms, body_terms)
            self.documents[doc_id] = {'title': title, 'body': body}

    def search(self, query):
        query = query.lower()
        
        # Check for phrase query (terms in quotes)
        if '"' in query:
            phrase = re.findall(r'"([^"]*)"', query)[0]
            phrase_terms = self.processor.process_text(phrase)
            return self.inverted_index.search_phrase(phrase_terms)
        
        # Normal boolean search
        operator = 'AND'
        if ' or ' in query:
            operator = 'OR'
            terms = query.split(' or ')
        elif ' and ' in query:
            terms = query.split(' and ')
        else:
            terms = query.split()
            
        terms = [term.strip() for term in terms]
        doc_ids = self.inverted_index.search(terms, operator=operator)
        
        # Rank results using VSM
        return self.inverted_index.rank_results(terms, doc_ids)


# ==== Initialize and Build Index ====
print("Loading dataset and building index...")
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
search_engine = SearchEngine(newsgroups_data.data)
search_engine.build_index()


# ==== Flask Routes ====
@app.route("/", methods=["GET", "POST"]) 
def index():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form["query"]
        ranked_docs = search_engine.search(query)  # Returns [(doc_id, score), ...]
        
        # Take top 5 results and get their documents
        for doc_id, score in list(ranked_docs)[:5]:
            doc = search_engine.documents[doc_id]
            results.append({
                "id": doc_id,
                "title": doc["title"],
                "snippet": doc["body"][:200].replace("\n", " ") + "...",
                "score": round(score, 3)  # Add score to results
            })

    return render_template("index.html", query=query, results=results)


@app.route("/document/<int:doc_id>")
def view_document(doc_id):
    if doc_id in search_engine.documents:
        return render_template("document.html", document=search_engine.documents[doc_id])
    return "Document not found", 404


if __name__ == "__main__":
    app.run(debug=True, port=5002)  # Changed port to 50002
