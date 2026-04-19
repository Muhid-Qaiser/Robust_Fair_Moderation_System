import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin, BaseEstimator

BLOCKLIST = {
    "direct_threat": [
        re.compile(r"\b(?:i will|i'll|i am going to|im gonna|someone should)\s+(kill|murder|shoot|stab|hurt)\s+you\b", re.IGNORECASE),
        re.compile(r"\b(?:im|i am)\s+coming\s+to\s+(kill|murder|shoot|stab|hurt)\s+you\b", re.IGNORECASE),
        re.compile(r"\b(?:i will|i'll)\s+find\s+where\s+you\s+live\b", re.IGNORECASE),
        re.compile(r"\byou(?:'re| are|\s+are)\s+(?:going to|gonna)\s+die\b", re.IGNORECASE),
        re.compile(r"\bhope\s+you\s+(?:die|get killed|get shot)\b", re.IGNORECASE)
    ],
    "self_harm_directed": [
        re.compile(r"\b(?:you\s+should|go|just)\s+(?:kill|hurt|hang)\s+yourself\b", re.IGNORECASE),
        re.compile(r"\bdo\s+(?:us|everyone)\s+a\s+favou?r\s+and\s+(?:die|disappear|kill yourself)\b", re.IGNORECASE),
        re.compile(r"\bnobody\s+would\s+(?:care|miss you)\s+if\s+you\s+(?:died|killed yourself)\b", re.IGNORECASE),
        re.compile(r"\bdrink\s+bleach\b", re.IGNORECASE)
    ],
    "doxxing_stalking": [
        re.compile(r"\bi\s+(?:know|found)\s+where\s+you\s+live\b", re.IGNORECASE),
        re.compile(r"\b(?:i will|i'll|im gonna|i am going to)\s+(?:post|leak|expose)\s+your\s+(?:address|location|real name|info)\b", re.IGNORECASE),
        re.compile(r"\bi\s+(?:know|found)\s+your\s+real\s+name\b", re.IGNORECASE),
        re.compile(r"\beveryone\s+will\s+know\s+who\s+you\s+really\s+are\b", re.IGNORECASE)
    ],
    "dehumanization": [
        re.compile(r"\b\w+\s+are\s+(?:not|less than|sub)\s+(?:human|people|person)s?\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+are\s+(?:just\s+|nothing\s+but\s+)?(?:animals|rats|pigs|monkeys|parasites|roaches)\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+should\s+be\s+(?:exterminated|eradicated|wiped out|destroyed)\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+are\s+a\s+(?:disease|cancer|virus|plague|infection)\b", re.IGNORECASE)
    ],
    "coordinated_harassment": [
        re.compile(r"\b(?:everyone|let's all|we should)\s+(?:go\s+after|report|raid|attack)(?=\s+(?:this|him|her|them|that|profile|account|user|channel))\b", re.IGNORECASE),
        re.compile(r"\bmass\s+report\s+(?:this|their)\s+(?:account|profile|user)\b", re.IGNORECASE),
        re.compile(r"\b(?:everyone|please)\s+report\s+@\w+\b", re.IGNORECASE)
    ]
}

def input_filter(text: str) -> dict | None:
    """Returns a block decision dict if matched, else None."""
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {"decision": "block", "layer": "input_filter", "category": category, "confidence": 1.0}
    return None

class HFPipelineWrapper(ClassifierMixin, BaseEstimator):
    """Wrapper to make HuggingFace models compatible with scikit-learn."""
    _estimator_type = "classifier"
    
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None

    def load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()

    def fit(self, X, y):
        self.load_model()
        self.classes_ = np.array([0, 1]) 
        return self

    def predict_proba(self, X):
        self.load_model()
        probs = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_texts = list(X[i:i+batch_size])
                inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_probs = torch.sigmoid(logits[:, 1]).cpu().numpy()
                p = np.vstack([1 - batch_probs, batch_probs]).T
                probs.append(p)
        return np.vstack(probs)
        
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

class ModerationPipeline:
    def __init__(self, model_path="./distilbert-toxic-mitigated"):
        self.filter = input_filter
        self.hf_wrapper = HFPipelineWrapper(model_path)
        
        # ADD THIS LINE: explicitly "fit" the pre-trained model so sklearn knows it has classes_
        self.hf_wrapper.fit(None, None)
        
        self.calibrator = CalibratedClassifierCV(estimator=self.hf_wrapper, method='isotonic', cv='prefit')
        self.is_calibrated = False

    def fit_calibrator(self, val_texts, val_labels):
        """Fit the probability calibrator using validation data."""
        self.calibrator.fit(val_texts, val_labels)
        self.is_calibrated = True

    def predict(self, text: str) -> dict:
        """Run the three layers in sequence and return a structured decision dictionary."""
        # Layer 1: Input filter
        filter_result = self.filter(text)
        if filter_result:
            return filter_result

        # Layer 2 & 3: Calibrated model
        if not self.is_calibrated:
            raise ValueError("Calibrator is not fitted. Call fit_calibrator() first.")
            
        prob = self.calibrator.predict_proba([text])[0][1]
        
        if prob >= 0.6:
            return {"decision": "block", "layer": "model", "confidence": float(prob)}
        elif prob <= 0.4:
            return {"decision": "allow", "layer": "model", "confidence": float(prob)}
        else:
            return {"decision": "review", "layer": "model", "confidence": float(prob)}
