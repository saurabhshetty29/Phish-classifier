"""
Simplified pipeline for testing without clustering dependencies.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from cache_store import ClassificationCache
from model_tfidf_lr import TFIDFLogisticClassifier
from parse_eml import EmailParser
from rules import RuleBasedClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplePhishingPipeline:
    """Simplified pipeline without clustering for testing."""
    
    def __init__(self, 
                 input_dir: str = "data/raw_eml",
                 output_dir: str = "outputs",
                 model_dir: str = "models",
                 cache_dir: str = "cache"):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.parser = EmailParser()
        self.rule_classifier = RuleBasedClassifier()
        self.ml_classifier = TFIDFLogisticClassifier(str(self.model_dir))
        self.cache = ClassificationCache(str(self.cache_dir / "classifications.db"))
        
        # Pipeline state
        self.parsed_emails = []
        self.rule_results = []
        self.ml_results = []
        self.final_labels = []
    
    def run_pipeline(self, train_ml: bool = True) -> Dict:
        """Run the simplified pipeline."""
        logger.info("Starting simplified phishing classification pipeline...")
        
        pipeline_stats = {}
        
        # Step 1: Parse emails
        logger.info("Step 1: Parsing emails...")
        parsed_file = self.output_dir / "parsed_emails.jsonl"
        count = self.parser.parse_eml_directory(self.input_dir, parsed_file)
        
        # Load parsed emails
        self.parsed_emails = []
        with open(parsed_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.parsed_emails.append(json.loads(line))
        
        pipeline_stats['parsed_emails'] = count
        logger.info(f"Parsed {count} emails successfully")
        
        if count == 0:
            logger.error("No emails to process")
            return pipeline_stats
        
        # Step 2: Rule classification
        logger.info("Step 2: Rule-based classification...")
        rule_results = self.rule_classifier.classify_batch(self.parsed_emails)
        self.rule_results = rule_results
        
        # Cache rule results
        self.cache.batch_set_classifications(rule_results)
        
        # Count by label
        rule_counts = {}
        for result in rule_results:
            label = result['label']
            rule_counts[label] = rule_counts.get(label, 0) + 1
        
        pipeline_stats['rule_classification'] = rule_counts
        logger.info(f"Rule classification complete: {rule_counts}")
        
        # Step 3: ML classification on UNKNOWN emails
        logger.info("Step 3: ML classification...")
        
        # Get UNKNOWN emails
        unknown_emails = []
        unknown_indices = []
        for i, result in enumerate(rule_results):
            if result['label'] == 'UNKNOWN':
                unknown_emails.append(self.parsed_emails[i])
                unknown_indices.append(i)
        
        if unknown_emails:
            if train_ml:
                logger.info("Training ML model...")
                # Use rule results for training
                training_emails = []
                training_labels = []
                for i, result in enumerate(rule_results):
                    if result['label'] in ['GOOD', 'BAD']:
                        training_emails.append(self.parsed_emails[i])
                        training_labels.append(result['label'])
                
                if training_emails and training_labels:
                    metrics = self.ml_classifier.train(training_emails, training_labels)
                    logger.info(f"ML model trained. Test accuracy: {metrics['test_accuracy']:.3f}")
                else:
                    logger.warning("No training data available")
                    return pipeline_stats
            else:
                try:
                    self.ml_classifier.load_model()
                except:
                    logger.error("No trained model available")
                    return pipeline_stats
            
            # Classify unknown emails
            auto_labels, ambiguous = self.ml_classifier.classify_split(unknown_emails)
            
            # Map back to original indices
            ml_results = []
            for i, (auto_label, email_idx) in enumerate(zip(auto_labels, unknown_indices)):
                auto_label['email_index'] = email_idx
                ml_results.append(auto_label)
            
            self.ml_results = ml_results
            
            # Cache ML results
            self.cache.batch_set_classifications(ml_results)
            
            # Count by label
            ml_counts = {}
            for result in ml_results:
                label = result['label']
                ml_counts[label] = ml_counts.get(label, 0) + 1
            
            pipeline_stats['ml_classification'] = ml_counts
            logger.info(f"ML classification complete: {ml_counts}")
        
        # Step 4: Generate final labels
        logger.info("Step 4: Generating final labels...")
        
        final_labels = []
        
        # Create mapping from email index to results
        email_results = {}
        
        # Add rule results
        for i, result in enumerate(rule_results):
            email_results[i] = [result]
        
        # Add ML results
        for ml_result in self.ml_results:
            email_idx = ml_result['email_index']
            if email_idx in email_results:
                email_results[email_idx].append(ml_result)
        
        # Generate final labels
        for i, email in enumerate(self.parsed_emails):
            fingerprint = email.get('fingerprint')
            
            if i in email_results:
                results = email_results[i]
                
                # Choose the result with highest confidence
                best_result = max(results, key=lambda x: x.get('confidence', 0))
                
                final_label = {
                    'fingerprint': fingerprint,
                    'label': best_result['label'],
                    'confidence': best_result['confidence'],
                    'source': best_result['source'],
                    'context': best_result.get('context', {}),
                    'all_sources': [r['source'] for r in results]
                }
            else:
                # Fallback
                final_label = {
                    'fingerprint': fingerprint,
                    'label': 'UNKNOWN',
                    'confidence': 0.0,
                    'source': 'none',
                    'context': {},
                    'all_sources': []
                }
            
            final_labels.append(final_label)
        
        self.final_labels = final_labels
        
        # Save final labels
        labels_file = self.output_dir / "labels.jsonl"
        with open(labels_file, 'w', encoding='utf-8') as f:
            for label in final_labels:
                f.write(json.dumps(label, ensure_ascii=False) + '\n')
        
        # Generate summary
        label_counts = {}
        source_counts = {}
        for label in final_labels:
            label_counts[label['label']] = label_counts.get(label['label'], 0) + 1
            source_counts[label['source']] = source_counts.get(label['source'], 0) + 1
        
        pipeline_stats['final_labels'] = len(final_labels)
        pipeline_stats['summary'] = {
            'total_emails': len(final_labels),
            'label_distribution': label_counts,
            'source_distribution': source_counts,
            'bad_percentage': (label_counts.get('BAD', 0) / len(final_labels)) * 100
        }
        
        logger.info("Pipeline complete!")
        logger.info(f"Final summary: {pipeline_stats['summary']}")
        
        return pipeline_stats


def main():
    """CLI interface for the simplified pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified Phishing Classification Pipeline")
    parser.add_argument("--input-dir", default="data/raw_eml", help="Directory containing .eml files")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory")
    parser.add_argument("--no-train", action="store_true", help="Skip ML model training")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SimplePhishingPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        cache_dir=args.cache_dir
    )
    
    # Run pipeline
    stats = pipeline.run_pipeline(train_ml=not args.no_train)
    print(f"Pipeline complete! Stats: {stats}")


if __name__ == "__main__":
    main()
