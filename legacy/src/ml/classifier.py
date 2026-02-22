"""Machine learning classifier for receipt type classification."""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class ReceiptClassifier:
    """ML-based receipt type classifier."""
    
    def __init__(self, algorithm: str = 'random_forest'):
        self.algorithm = algorithm
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.is_trained = False
        self._initialize()
    
    def _initialize(self):
        """Initialize vectorizer and classifier."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import LabelEncoder
            
            self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            self.label_encoder = LabelEncoder()
            self.classifier = self._create_classifier()
        except ImportError:
            logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
            raise
    
    def _create_classifier(self):
        """Create classifier based on algorithm choice."""
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        
        classifiers = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42
            ),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        }
        
        return classifiers.get(self.algorithm, RandomForestClassifier())
    
    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """Train the classifier."""
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import accuracy_score, classification_report
        
        logger.info(f"Training {self.algorithm} classifier...")
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.classifier, X_train, y_train, cv=5
        )
        
        self.is_trained = True
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
        }
        
        logger.info(f"Training completed. Accuracy: {accuracy:.2%}")
        return results
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict receipt type."""
        from ..utils.models import ReceiptType
        
        if not self.is_trained:
            return ReceiptType.UNKNOWN.value, 0.0
        
        # Vectorize
        X = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = float(max(probabilities))
        
        # Decode label
        label = self.label_encoder.inverse_transform([prediction])[0]
        
        return label, confidence
    
    def save(self, classifier_path: Path, vectorizer_path: Path):
        """Save model to disk."""
        with open(classifier_path, 'wb') as f:
            pickle.dump((self.classifier, self.label_encoder, self.is_trained), f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info(f"Model saved to {classifier_path}")
    
    def load(self, classifier_path: Path, vectorizer_path: Path):
        """Load model from disk."""
        if classifier_path.exists() and vectorizer_path.exists():
            with open(classifier_path, 'rb') as f:
                self.classifier, self.label_encoder, self.is_trained = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            logger.info(f"Model loaded from {classifier_path}")
            return True
        return False
