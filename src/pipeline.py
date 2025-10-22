"""
Main pipeline orchestrator for the phishing classification system.

This module coordinates all stages of the hybrid classification pipeline:
1. Parse .eml files
2. Apply rule-based filters
3. Run ML classifier on ambiguous emails
4. Cluster remaining emails
5. Use LLM for final classification
6. Cache results and generate final labels
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tqdm

from cache_store import ClassificationCache
from dedupe_cluster import EmailClusterer
from llm_batch import LLMBatchClassifier
from model_tfidf_lr import TFIDFLogisticClassifier
from parse_eml import EmailParser
from rules import RuleBasedClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhishingClassificationPipeline:
    """Main orchestrator for the phishing classification pipeline."""
    
    def __init__(self, 
                 input_dir: str = "data/raw_eml",
                 output_dir: str = "outputs",
                 model_dir: str = "models",
                 cache_dir: str = "cache",
                 openai_api_key: Optional[str] = None):
        
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
        self.clusterer = EmailClusterer()
        self.llm_classifier = LLMBatchClassifier(api_key=openai_api_key)
        self.cache = ClassificationCache(str(self.cache_dir / "classifications.db"))
        
        # Pipeline state
        self.parsed_emails = []
        self.rule_results = []
        self.ml_results = []
        self.ambiguous_emails = []
        self.cluster_results = {}
        self.llm_results = []
        self.final_labels = []
    
    def step1_parse_emails(self) -> int:
        """Step 1: Parse .eml files into structured data."""
        logger.info("Step 1: Parsing .eml files...")
        
        parsed_file = self.output_dir / "parsed_emails.jsonl"
        count = self.parser.parse_eml_directory(self.input_dir, parsed_file)
        
        # Load parsed emails
        self.parsed_emails = []
        with open(parsed_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.parsed_emails.append(json.loads(line))
        
        logger.info(f"Parsed {count} emails successfully")
        return count
    
    def step2_rule_classification(self) -> Dict[str, int]:
        """Step 2: Apply rule-based classification."""
        logger.info("Step 2: Applying rule-based classification...")
        
        # Check cache first
        cached_count = 0
        uncached_emails = []
        
        for email in self.parsed_emails:
            fingerprint = email.get('fingerprint')
            cached_result = self.cache.get_classification(fingerprint)
            
            if cached_result and cached_result['source'] == 'rules':
                self.rule_results.append(cached_result)
                cached_count += 1
            else:
                uncached_emails.append(email)
        
        # Classify uncached emails
        if uncached_emails:
            new_results = self.rule_classifier.classify_batch(uncached_emails)
            self.rule_results.extend(new_results)
            
            # Cache new results
            self.cache.batch_set_classifications(new_results)
        
        # Count by label
        label_counts = {}
        for result in self.rule_results:
            label = result['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info(f"Rule classification complete: {label_counts}")
        logger.info(f"Used {cached_count} cached results, classified {len(uncached_emails)} new emails")
        
        return label_counts
    
    def step3_ml_classification(self, train: bool = False) -> Dict[str, int]:
        """Step 3: Apply ML classification to ambiguous emails."""
        logger.info("Step 3: Applying ML classification...")
        
        # Get emails that were UNKNOWN from rules
        unknown_emails = []
        for i, result in enumerate(self.rule_results):
            if result['label'] == 'UNKNOWN':
                unknown_emails.append(self.parsed_emails[i])
        
        if not unknown_emails:
            logger.info("No ambiguous emails for ML classification")
            return {}
        
        # Train model if requested
        if train:
            logger.info("Training ML model...")
            # Get high-confidence labels from cache for training
            training_emails, training_labels = self.cache.get_training_data(min_confidence=0.8)
            
            if training_emails and training_labels:
                metrics = self.ml_classifier.train(training_emails, training_labels)
                logger.info(f"ML model trained. Test accuracy: {metrics['test_accuracy']:.3f}")
            else:
                logger.warning("No training data available, using pre-trained model")
                try:
                    self.ml_classifier.load_model()
                except:
                    logger.error("No pre-trained model available")
                    return {}
        else:
            # Load existing model
            try:
                self.ml_classifier.load_model()
            except:
                logger.error("No trained model available. Use --train flag to train first.")
                return {}
        
        # Classify ambiguous emails
        auto_labels, ambiguous = self.ml_classifier.classify_split(unknown_emails)
        
        # Update results
        self.ml_results = auto_labels
        self.ambiguous_emails = ambiguous
        
        # Cache ML results
        self.cache.batch_set_classifications(auto_labels)
        
        # Count by label
        label_counts = {}
        for result in auto_labels:
            label = result['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info(f"ML classification complete: {label_counts}")
        logger.info(f"Auto-labeled {len(auto_labels)} emails, {len(ambiguous)} remain ambiguous")
        
        return label_counts
    
    def step4_clustering(self) -> Dict:
        """Step 4: Cluster ambiguous emails for LLM processing."""
        logger.info("Step 4: Clustering ambiguous emails...")
        
        if not self.ambiguous_emails:
            logger.info("No ambiguous emails to cluster")
            return {}
        
        # Cluster emails
        representatives, cluster_mappings = self.clusterer.cluster_emails(
            self.ambiguous_emails, min_cluster_size=2
        )
        
        self.cluster_results = {
            'representatives': representatives,
            'cluster_mappings': cluster_mappings
        }
        
        # Save clustering results
        cluster_file = self.output_dir / "clustering_results.json"
        self.clusterer.save_clustering_results(cluster_file, self.ambiguous_emails, 
                                              representatives, cluster_mappings)
        
        logger.info(f"Clustering complete: {len(representatives)} clusters found")
        
        return self.cluster_results
    
    def step5_llm_classification(self) -> Dict[str, int]:
        """Step 5: Use LLM for final classification of representatives."""
        logger.info("Step 5: LLM classification of representatives...")
        
        if not self.cluster_results:
            logger.info("No clusters available for LLM classification")
            return {}
        
        # Get representative emails
        representatives = self.cluster_results['representatives']
        cluster_mappings = self.cluster_results['cluster_mappings']
        
        rep_emails = [self.ambiguous_emails[rep_idx] for rep_idx in representatives.values()]
        
        # Classify representatives
        llm_results = self.llm_classifier.classify_representatives(rep_emails)
        
        # Propagate labels to all emails in clusters
        propagated_results = self.llm_classifier.propagate_labels(
            llm_results, cluster_mappings, representatives
        )
        
        self.llm_results = propagated_results
        
        # Cache LLM results
        llm_cache_data = []
        for result in propagated_results:
            email_idx = result['email_index']
            email = self.ambiguous_emails[email_idx]
            fingerprint = email.get('fingerprint')
            
            llm_cache_data.append({
                'fingerprint': fingerprint,
                'label': result['label'],
                'confidence': result['confidence'],
                'source': 'llm',
                'context': {'reason': result['reason']}
            })
        
        self.cache.batch_set_classifications(llm_cache_data)
        
        # Count by label
        label_counts = {}
        for result in propagated_results:
            label = result['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info(f"LLM classification complete: {label_counts}")
        
        return label_counts
    
    def step6_generate_final_labels(self) -> List[Dict]:
        """Step 6: Generate final labels combining all sources."""
        logger.info("Step 6: Generating final labels...")
        
        final_labels = []
        
        # Create mapping from email index to all results
        email_results = {}
        
        # Add rule results
        for i, result in enumerate(self.rule_results):
            if i not in email_results:
                email_results[i] = []
            email_results[i].append(result)
        
        # Add ML results (these are for UNKNOWN emails from rules)
        unknown_indices = [i for i, result in enumerate(self.rule_results) 
                          if result['label'] == 'UNKNOWN']
        
        for ml_result in self.ml_results:
            # Find corresponding email index
            for i, unknown_idx in enumerate(unknown_indices):
                if i < len(self.ambiguous_emails):
                    ambiguous_idx = i
                    if ambiguous_idx not in email_results:
                        email_results[ambiguous_idx] = []
                    email_results[ambiguous_idx].append(ml_result)
        
        # Add LLM results
        for llm_result in self.llm_results:
            email_idx = llm_result['email_index']
            if email_idx not in email_results:
                email_results[email_idx] = []
            email_results[email_idx].append(llm_result)
        
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
                # Fallback for emails without any classification
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
        
        logger.info(f"Final labels generated: {label_counts}")
        logger.info(f"By source: {source_counts}")
        
        return final_labels
    
    def run_full_pipeline(self, train_ml: bool = False, use_llm: bool = True) -> Dict:
        """Run the complete classification pipeline."""
        logger.info("Starting full phishing classification pipeline...")
        
        pipeline_stats = {}
        
        # Step 1: Parse emails
        parsed_count = self.step1_parse_emails()
        pipeline_stats['parsed_emails'] = parsed_count
        
        if parsed_count == 0:
            logger.error("No emails to process")
            return pipeline_stats
        
        # Step 2: Rule classification
        rule_counts = self.step2_rule_classification()
        pipeline_stats['rule_classification'] = rule_counts
        
        # Step 3: ML classification
        ml_counts = self.step3_ml_classification(train=train_ml)
        pipeline_stats['ml_classification'] = ml_counts
        
        # Step 4: Clustering (if there are ambiguous emails)
        if self.ambiguous_emails:
            cluster_results = self.step4_clustering()
            pipeline_stats['clustering'] = {
                'clusters': len(cluster_results.get('representatives', {})),
                'ambiguous_emails': len(self.ambiguous_emails)
            }
            
            # Step 5: LLM classification (if enabled)
            if use_llm:
                llm_counts = self.step5_llm_classification()
                pipeline_stats['llm_classification'] = llm_counts
            else:
                logger.info("Skipping LLM classification")
        
        # Step 6: Generate final labels
        final_labels = self.step6_generate_final_labels()
        pipeline_stats['final_labels'] = len(final_labels)
        
        # Generate final summary
        final_summary = self.generate_summary()
        pipeline_stats['summary'] = final_summary
        
        logger.info("Pipeline complete!")
        logger.info(f"Final summary: {final_summary}")
        
        return pipeline_stats
    
    def generate_summary(self) -> Dict:
        """Generate a summary of the classification results."""
        if not self.final_labels:
            return {}
        
        # Count by label
        label_counts = {}
        confidence_by_label = {}
        source_counts = {}
        
        for label in self.final_labels:
            label_name = label['label']
            confidence = label['confidence']
            source = label['source']
            
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
            
            if label_name not in confidence_by_label:
                confidence_by_label[label_name] = []
            confidence_by_label[label_name].append(confidence)
            
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Calculate average confidence by label
        avg_confidence = {}
        for label, confidences in confidence_by_label.items():
            avg_confidence[label] = sum(confidences) / len(confidences)
        
        return {
            'total_emails': len(self.final_labels),
            'label_distribution': label_counts,
            'average_confidence': avg_confidence,
            'source_distribution': source_counts,
            'bad_percentage': (label_counts.get('BAD', 0) / len(self.final_labels)) * 100
        }


def main():
    """CLI interface for the phishing classification pipeline."""
    parser = argparse.ArgumentParser(description="Phishing Classification Pipeline")
    
    # Input/Output
    parser.add_argument("--input-dir", default="data/raw_eml", help="Directory containing .eml files")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory")
    
    # Pipeline options
    parser.add_argument("--train", action="store_true", help="Train ML model")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM classification")
    parser.add_argument("--openai-api-key", help="OpenAI API key for LLM classification")
    
    # Individual steps
    parser.add_argument("--step", choices=['1', '2', '3', '4', '5', '6', 'all'], 
                       default='all', help="Run specific pipeline step")
    
    # Cache management
    parser.add_argument("--clear-cache", action="store_true", help="Clear old cache entries")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PhishingClassificationPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        cache_dir=args.cache_dir,
        openai_api_key=args.openai_api_key
    )
    
    # Handle cache operations
    if args.clear_cache:
        count = pipeline.cache.clear_old_classifications(days=30)
        print(f"Cleared {count} old cache entries")
        return
    
    if args.cache_stats:
        stats = pipeline.cache.get_cache_stats()
        print("Cache Statistics:")
        print(f"  Total classifications: {stats['total_classifications']}")
        print(f"  Recent (24h): {stats['recent_24h']}")
        print("  By source:")
        for source, count in stats['by_source'].items():
            print(f"    {source}: {count}")
        print("  By label:")
        for label, count in stats['by_label'].items():
            print(f"    {label}: {count}")
        return
    
    # Run pipeline steps
    if args.step == 'all':
        stats = pipeline.run_full_pipeline(
            train_ml=args.train,
            use_llm=not args.no_llm
        )
        print(f"Pipeline complete! Stats: {stats}")
    
    elif args.step == '1':
        count = pipeline.step1_parse_emails()
        print(f"Parsed {count} emails")
    
    elif args.step == '2':
        counts = pipeline.step2_rule_classification()
        print(f"Rule classification: {counts}")
    
    elif args.step == '3':
        counts = pipeline.step3_ml_classification(train=args.train)
        print(f"ML classification: {counts}")
    
    elif args.step == '4':
        results = pipeline.step4_clustering()
        print(f"Clustering: {len(results.get('representatives', {}))} clusters")
    
    elif args.step == '5':
        counts = pipeline.step5_llm_classification()
        print(f"LLM classification: {counts}")
    
    elif args.step == '6':
        labels = pipeline.step6_generate_final_labels()
        print(f"Generated {len(labels)} final labels")


if __name__ == "__main__":
    main()
