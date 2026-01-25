import os
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from django.conf import settings


class SymptomClassifier:
    
    _instance = None
    _model = None
    _tokenizer = None
    _classifier = None
    _id2label = None
    _label2id = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        model_path = settings.MODEL_PATH
        
        print(f"Loading model from: {model_path}")
        
        # Load label mappings
        with open(os.path.join(model_path, 'label2id.json'), 'r') as f:
            self._label2id = json.load(f)
        
        with open(os.path.join(model_path, 'id2label.json'), 'r') as f:
            self._id2label = {int(k): v for k, v in json.load(f).items()}
        
        # Determine device
        device = 0 if torch.cuda.is_available() else -1
        
        # Create pipeline
        self._classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=device,
            top_k=5
        )
        
        print("Model loaded successfully!")

    def predict(self, symptoms_text):
       
        if not symptoms_text.startswith("Patient presents with:"):
            symptoms_text = f"Patient presents with: {symptoms_text}"
        
        results = self._classifier(symptoms_text)[0]
        
        top_5 = [
            {
                'condition': result['label'],
                'confidence': round(result['score'] * 100, 2)
            }
            for result in results
        ]
        
        return {
            'predicted_condition': top_5[0]['condition'],
            'confidence': top_5[0]['confidence'],
            'top_5_predictions': top_5
        }

    def get_all_conditions(self):
        return list(self._label2id.keys())


# Singleton instance
def get_classifier():
    return SymptomClassifier()