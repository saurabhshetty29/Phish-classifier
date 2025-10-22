"""
Lightweight ML classifier using TF-IDF and Logistic Regression.

This module implements a fast, cheap ML model for high-confidence predictions
on emails that couldn't be classified by rules.
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFIDFLogisticClassifier:
    """Lightweight ML classifier using TF-IDF + Logistic Regression."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        self.is_trained = False
        self.feature_names = []
    
    def extract_features(self, email_data: Dict) -> Dict[str, float]:
        """Extract numerical features from email data."""
        features = {}
        
        # Text features
        subject = email_data.get('subject', '')
        snippet = email_data.get('snippet', '')
        body = email_data.get('body', '')
        
        features['subject_length'] = len(subject)
        features['snippet_length'] = len(snippet)
        features['body_length'] = len(body)
        
        # URL features
        urls = email_data.get('urls', [])
        url_domains = email_data.get('url_domains', [])
        
        features['url_count'] = len(urls)
        features['url_domain_count'] = len(url_domains)
        features['has_urls'] = 1.0 if urls else 0.0
        
        # Authentication features
        features['spf_pass'] = 1.0 if email_data.get('spf_pass', False) else 0.0
        features['dkim_pass'] = 1.0 if email_data.get('dkim_pass', False) else 0.0
        features['dmarc_pass'] = 1.0 if email_data.get('dmarc_pass', False) else 0.0
        features['auth_score'] = sum([
            email_data.get('spf_pass', False),
            email_data.get('dkim_pass', False),
            email_data.get('dmarc_pass', False)
        ])
        
        # Content features
        features['has_html'] = 1.0 if email_data.get('has_html', False) else 0.0
        features['content_ratio'] = len(snippet) / max(len(body), 1)
        
        # Sender domain features
        sender_domain = email_data.get('sender_domain', '')
        features['domain_length'] = len(sender_domain)
        features['has_domain'] = 1.0 if sender_domain else 0.0
        
        # Suspicious patterns
        text_content = f"{subject} {snippet}".lower()
        
        # Count suspicious keywords
        suspicious_keywords = [
            'urgent', 'verify', 'account', 'password', 'click', 'here',
            'immediately', 'suspended', 'expired', 'action', 'required',
            'confirm', 'update', 'billing', 'payment', 'invoice'
        ]
        
        features['suspicious_keyword_count'] = sum(
            1 for keyword in suspicious_keywords if keyword in text_content
        )
        
        # Count exclamation marks and caps
        features['exclamation_count'] = text_content.count('!')
        features['caps_ratio'] = sum(1 for c in text_content if c.isupper()) / max(len(text_content), 1)
        
        # Count numbers (potential phone numbers, amounts)
        features['number_count'] = len(re.findall(r'\d+', text_content))
        
        return features
    
    def prepare_training_data(self, emails: List[Dict], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for the model."""
        # Filter out UNKNOWN labels for training
        training_data = []
        training_labels = []
        
        for email, label in zip(emails, labels):
            if label in ['GOOD', 'BAD']:
                training_data.append(email)
                training_labels.append(1 if label == 'BAD' else 0)
        
        if not training_data:
            raise ValueError("No training data with GOOD/BAD labels found")
        
        logger.info(f"Prepared {len(training_data)} training samples")
        
        # Extract text features
        text_data = []
        for email in training_data:
            subject = email.get('subject', '')
            snippet = email.get('snippet', '')
            text_data.append(f"{subject} {snippet}")
        
        # Vectorize text
        X_text = self.vectorizer.fit_transform(text_data)
        
        # Extract numerical features
        numerical_features = []
        for email in training_data:
            features = self.extract_features(email)
            numerical_features.append(list(features.values()))
        
        X_numerical = np.array(numerical_features)
        
        # Combine text and numerical features
        X_combined = np.hstack([X_text.toarray(), X_numerical])
        
        y = np.array(training_labels)
        
        return X_combined, y
    
    def train(self, emails: List[Dict], labels: List[str], test_size: float = 0.2) -> Dict:
        """Train the TF-IDF + Logistic Regression model."""
        logger.info("Training TF-IDF + Logistic Regression model...")
        
        # Prepare training data
        X, y = self.prepare_training_data(emails, labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.is_trained = True
        
        # Save model components
        self.save_model()
        
        logger.info(f"Model training complete. Test accuracy: {test_score:.3f}")
        
        return metrics
    
    def predict_proba(self, email_data: Dict) -> Tuple[float, float]:
        """Predict probability of email being BAD (phishing)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract text features
        subject = email_data.get('subject', '')
        snippet = email_data.get('snippet', '')
        text_content = f"{subject} {snippet}"
        
        # Vectorize text
        X_text = self.vectorizer.transform([text_content])
        
        # Extract numerical features
        features = self.extract_features(email_data)
        X_numerical = np.array([list(features.values())])
        
        # Combine features
        X_combined = np.hstack([X_text.toarray(), X_numerical])
        
        # Scale features
        X_scaled = self.scaler.transform(X_combined)
        
        # Predict probabilities
        proba = self.model.predict_proba(X_scaled)[0]
        
        return proba[0], proba[1]  # prob_good, prob_bad
    
    def classify_split(self, emails: List[Dict], p_hi: float = 0.95, p_lo: float = 0.05) -> Tuple[List[Dict], List[Dict]]:
        """Classify emails and split into high-confidence and ambiguous groups."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        auto_labels = []
        ambiguous = []
        
        for email in emails:
            try:
                prob_good, prob_bad = self.predict_proba(email)
                confidence = max(prob_good, prob_bad)
                predicted_label = 'BAD' if prob_bad > prob_good else 'GOOD'
                
                result = {
                    'fingerprint': email.get('fingerprint'),
                    'label': predicted_label,
                    'confidence': confidence,
                    'source': 'ml',
                    'prob_good': prob_good,
                    'prob_bad': prob_bad
                }
                
                if confidence >= p_hi:
                    auto_labels.append(result)
                elif confidence <= p_lo:
                    # Low confidence predictions go to ambiguous for LLM review
                    ambiguous.append(email)
                else:
                    # Medium confidence - could go either way
                    ambiguous.append(email)
                    
            except Exception as e:
                logger.error(f"Failed to classify email {email.get('fingerprint', 'unknown')}: {e}")
                ambiguous.append(email)
        
        logger.info(f"ML classification: {len(auto_labels)} auto-labeled, {len(ambiguous)} ambiguous")
        
        return auto_labels, ambiguous
    
    def save_model(self):
        """Save the trained model components."""
        # Save vectorizer
        with open(self.model_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save scaler
        with open(self.model_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save model
        with open(self.model_dir / 'lr_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"Model saved to {self.model_dir}")
    
    def load_model(self):
        """Load the trained model components."""
        try:
            # Load vectorizer
            with open(self.model_dir / 'tfidf_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load scaler
            with open(self.model_dir / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load model
            with open(self.model_dir / 'lr_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            self.is_trained = True
            logger.info(f"Model loaded from {self.model_dir}")
            
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


def main():
    """CLI interface for ML model training and classification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression classifier")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--classify", help="Classify emails from JSONL file")
    parser.add_argument("--emails", help="Input JSONL file with emails")
    parser.add_argument("--labels", help="Input JSONL file with labels")
    parser.add_argument("--output", help="Output file for classifications")
    
    args = parser.parse_args()
    
    classifier = TFIDFLogisticClassifier()
    
    if args.train:
        if not args.emails or not args.labels:
            print("Error: --emails and --labels required for training")
            return
        
        # Load training data
        emails = []
        with open(args.emails, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    emails.append(json.loads(line))
        
        labels = []
        with open(args.labels, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    labels.append(json.loads(line)['label'])
        
        # Train model
        metrics = classifier.train(emails, labels)
        print(f"Training complete. Test accuracy: {metrics['test_accuracy']:.3f}")
    
    elif args.classify:
        if not args.output:
            print("Error: --output required for classification")
            return
        
        # Load model
        classifier.load_model()
        
        # Load emails
        emails = []
        with open(args.classify, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    emails.append(json.loads(line))
        
        # Classify
        auto_labels, ambiguous = classifier.classify_split(emails)
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            for result in auto_labels:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"Classification complete: {len(auto_labels)} auto-labeled, {len(ambiguous)} ambiguous")


if __name__ == "__main__":
    main()
