from entity_recognition import EntityRecognizer
from knowledge_base import KnowledgeBase
from answer_ranker import BERTAnswerRanker
import argparse

class QASystem:
    def __init__(self):
        print("Initializing QA System components...")
        self.entity_recognizer = EntityRecognizer()
        self.knowledge_base = KnowledgeBase()
        self.answer_ranker = BERTAnswerRanker()
        
    def answer_question(self, question: str) -> dict:
        """Process a question and return the best answer"""
        print(f"Processing question: {question}")
        
        # Step 1: Extract entities from the question
        entities = self.entity_recognizer.extract_entities(question)
        print(f"Identified entities: {entities}")
        
        # Step 2: Create search query using entities
        query = question
        if entities:
            # Enhance query with entity information
            entity_texts = [e["text"] for e in entities]
            query = query + " " + " ".join(entity_texts)
        
        # Step 3: Retrieve relevant documents
        documents = self.knowledge_base.retrieve_relevant_documents(query, top_k=5)
        print(f"Retrieved {len(documents)} relevant documents")
        
        # Step 4: Extract and rank answers using BERT
        if documents:
            ranked_answers = self.answer_ranker.extract_answers(question, documents)
            
            if ranked_answers:
                return {
                    "question": question,
                    "answer": ranked_answers[0]["answer"],
                    "confidence": ranked_answers[0]["confidence"],
                    "source": ranked_answers[0]["context_title"]
                }
        
        return {
            "question": question,
            "answer": "I couldn't find an answer to that question.",
            "confidence": 0.0,
            "source": None
        }

def main():
    parser = argparse.ArgumentParser(description="Question Answering System")
    parser.add_argument("--question", type=str, help="Question to answer")
    args = parser.parse_args()
    
    qa_system = QASystem()
    
    if args.question:
        result = qa_system.answer_question(args.question)
        print("\nAnswer:", result["answer"])
        print("Confidence:", result["confidence"])
        if result["source"]:
            print("Source:", result["source"])
    else:
        print("Interactive mode. Type 'exit' to quit.")
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == "exit":
                break
            
            result = qa_system.answer_question(question)
            print("\nAnswer:", result["answer"])
            print("Confidence:", result["confidence"])
            if result["source"]:
                print("Source:", result["source"])

if __name__ == "__main__":
    main()