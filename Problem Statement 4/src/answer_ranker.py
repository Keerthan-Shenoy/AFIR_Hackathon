import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
from typing import List, Dict, Tuple

class BERTAnswerRanker:
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def extract_answers(self, question: str, contexts: List[Dict]) -> List[Dict]:
        """Extract and rank answers from the given contexts using BERT"""
        answers = []
        
        for context_data in contexts:
            context = context_data["content"]
            
            # Tokenize and get model inputs
            inputs = self.tokenizer(
                question, 
                context, 
                add_special_tokens=True,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get answer span predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get start and end positions
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            # Find the best answer span
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores) + 1
            
            if answer_end <= answer_start:
                # Invalid span, skip
                continue
            
            # Convert tokens to actual text span
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            answer = self.tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end])
            
            # Calculate confidence score
            confidence = float(torch.max(start_scores) + torch.max(end_scores))
            
            if answer and not answer.startswith("[CLS]") and not answer.startswith("[SEP]"):
                answers.append({
                    "answer": answer,
                    "context_title": context_data["title"],
                    "confidence": confidence,
                    "source_id": context_data["id"]
                })
        
        # Sort by confidence
        answers = sorted(answers, key=lambda x: x["confidence"], reverse=True)
        return answers