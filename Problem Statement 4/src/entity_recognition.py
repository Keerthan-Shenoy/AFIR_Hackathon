import spacy
from typing import List, Dict, Tuple

class EntityRecognizer:
    def __init__(self):
        # Load spaCy English model with NER capabilities
        self.nlp = spacy.load("en_core_web_md")
        
    def extract_entities(self, question: str) -> List[Dict[str, str]]:
        """Extract named entities from a question"""
        doc = self.nlp(question)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "type": ent.label_
            })
        
        return entities
    
    def extract_question_type(self, question: str) -> str:
        """Determine the type of question (who, what, where, etc.)"""
        question_lower = question.lower().strip()
        
        if question_lower.startswith("who"):
            return "PERSON"
        elif question_lower.startswith("where"):
            return "LOCATION"
        elif question_lower.startswith("when"):
            return "DATE"
        elif question_lower.startswith("how many") or question_lower.startswith("how much"):
            return "QUANTITY"
        else:
            return "INFORMATION"