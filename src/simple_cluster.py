"""
Simplified clustering using TF-IDF and KMeans (no sentence-transformers dependency).

This version uses scikit-learn's built-in TfidfVectorizer for embeddings
instead of sentence-transformers to avoid dependency conflicts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleEmailClusterer:
    """Clusters similar emails using TF-IDF + KMeans."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.embeddings = None
        self.cluster_labels = None
        
    def prepare_text_for_embedding(self, email_data: Dict) -> str:
        """Prepare email text for clustering."""
        subject = email_data.get('subject', '')
        snippet = email_data.get('snippet', '')
        sender_domain = email_data.get('sender_domain', '')
        
        text_parts = []
        
        if subject:
            text_parts.append(f"{subject}")
        
        if snippet:
            text_parts.append(f"{snippet}")
        
        if sender_domain:
            text_parts.append(f"{sender_domain}")
        
        return " ".join(text_parts)
    
    def cluster_emails(self, emails: List[Dict], n_clusters: int = None) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """Cluster emails and return representatives and cluster mappings."""
        if not emails:
            return {}, {}
        
        logger.info(f"Clustering {len(emails)} emails...")
        
        # Prepare texts
        texts = [self.prepare_text_for_embedding(email) for email in emails]
        
        # Generate TF-IDF embeddings
        logger.info("Generating TF-IDF embeddings...")
        self.embeddings = self.vectorizer.fit_transform(texts).toarray()
        logger.info(f"Generated embeddings with shape: {self.embeddings.shape}")
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = min(max(2, len(emails) // 10), 100)
        
        logger.info(f"Using KMeans with {n_clusters} clusters...")
        
        # Cluster embeddings
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        cluster_centers = kmeans.cluster_centers_
        
        logger.info(f"KMeans clustering complete: {n_clusters} clusters found")
        
        # Find representative for each cluster
        representatives = {}
        cluster_mappings = {}
        
        # Group emails by cluster
        clusters = {}
        for i, label in enumerate(self.cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # Find representative for each cluster (closest to center)
        for cluster_id in clusters:
            cluster_items = clusters[cluster_id]
            cluster_embeddings = self.embeddings[cluster_items]
            
            # Calculate distances to center
            center = cluster_centers[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            representative_idx = cluster_items[np.argmin(distances)]
            
            representatives[cluster_id] = representative_idx
            cluster_mappings[cluster_id] = cluster_items
        
        logger.info(f"Found {len(representatives)} cluster representatives")
        
        return representatives, cluster_mappings
    
    def save_clustering_results(self, output_file: Path, emails: List[Dict], 
                              representatives: Dict[int, int], cluster_mappings: Dict[int, List[int]]):
        """Save clustering results to file."""
        results = {
            'representatives': {str(k): int(v) for k, v in representatives.items()},
            'cluster_mappings': {str(k): [int(x) for x in v] for k, v in cluster_mappings.items()},
            'cluster_summary': {},
            'total_emails': len(emails),
            'num_clusters': len(representatives)
        }
        
        # Add cluster summaries
        for cluster_id, email_indices in cluster_mappings.items():
            representative_idx = representatives[cluster_id]
            rep_email = emails[representative_idx]
            
            results['cluster_summary'][str(cluster_id)] = {
                'size': len(email_indices),
                'representative_index': int(representative_idx),
                'representative_fingerprint': rep_email.get('fingerprint'),
                'representative_subject': rep_email.get('subject'),
                'representative_from': rep_email.get('from'),
                'member_indices': [int(x) for x in email_indices[:10]]  # First 10 for preview
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Clustering results saved to {output_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"CLUSTERING SUMMARY")
        print(f"{'='*60}")
        print(f"Total emails clustered: {len(emails)}")
        print(f"Number of clusters: {len(representatives)}")
        print(f"Average cluster size: {len(emails) / len(representatives):.1f}")
        print(f"LLM calls needed: {len(representatives)} (vs {len(emails)} without clustering)")
        print(f"Cost reduction: {(1 - len(representatives)/len(emails))*100:.1f}%")
        print(f"{'='*60}\n")
        
        # Show top 5 largest clusters
        cluster_sizes = [(cid, len(members)) for cid, members in cluster_mappings.items()]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 5 Largest Clusters:")
        print("-" * 60)
        for i, (cluster_id, size) in enumerate(cluster_sizes[:5], 1):
            rep_idx = representatives[cluster_id]
            rep_email = emails[rep_idx]
            print(f"{i}. Cluster {cluster_id}: {size} emails")
            print(f"   Representative: {rep_email.get('subject', 'No subject')[:60]}")
            print(f"   From: {rep_email.get('from', 'Unknown')[:60]}")
            print()


def main():
    """CLI interface for simple clustering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple email clustering using TF-IDF")
    parser.add_argument("input_file", help="Input JSONL file with emails")
    parser.add_argument("output_file", help="Output JSON file with clustering results")
    parser.add_argument("--n-clusters", type=int, help="Number of clusters (default: auto)")
    
    args = parser.parse_args()
    
    # Load emails
    emails = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                emails.append(json.loads(line))
    
    print(f"Loaded {len(emails)} emails from {args.input_file}")
    
    # Cluster emails
    clusterer = SimpleEmailClusterer()
    representatives, cluster_mappings = clusterer.cluster_emails(emails, args.n_clusters)
    
    # Save results
    clusterer.save_clustering_results(Path(args.output_file), emails, representatives, cluster_mappings)


if __name__ == "__main__":
    main()

