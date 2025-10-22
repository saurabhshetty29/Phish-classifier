"""
Email deduplication and clustering module using embeddings.

This module uses sentence transformers to create embeddings and cluster
similar emails to reduce the number of LLM calls needed.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("HDBSCAN not available, will use KMeans as fallback")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailClusterer:
    """Clusters similar emails using embeddings to reduce LLM calls."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.cluster_labels = None
        self.cluster_centers = None
        
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
    
    def prepare_text_for_embedding(self, email_data: Dict) -> str:
        """Prepare email text for embedding generation."""
        subject = email_data.get('subject', '')
        snippet = email_data.get('snippet', '')
        sender_domain = email_data.get('sender_domain', '')
        
        # Combine key text elements
        text_parts = []
        
        if subject:
            text_parts.append(f"Subject: {subject}")
        
        if snippet:
            text_parts.append(f"Content: {snippet}")
        
        if sender_domain:
            text_parts.append(f"From: {sender_domain}")
        
        # Add URL domains for context
        url_domains = email_data.get('url_domains', [])
        if url_domains:
            text_parts.append(f"URLs: {', '.join(url_domains[:5])}")  # Limit to first 5 URLs
        
        return " | ".join(text_parts)
    
    def generate_embeddings(self, emails: List[Dict]) -> np.ndarray:
        """Generate embeddings for a list of emails."""
        self.load_model()
        
        # Prepare texts
        texts = [self.prepare_text_for_embedding(email) for email in emails]
        
        logger.info(f"Generating embeddings for {len(texts)} emails...")
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    def cluster_embeddings(self, embeddings: np.ndarray, min_cluster_size: int = 2, 
                          min_samples: int = 1) -> np.ndarray:
        """Cluster embeddings using HDBSCAN or KMeans fallback."""
        n_samples = len(embeddings)
        
        if n_samples < 2:
            return np.zeros(n_samples, dtype=int)
        
        # Determine optimal number of clusters for KMeans fallback
        max_clusters = min(20, n_samples // 2)
        min_clusters = max(2, n_samples // 10)
        
        if HDBSCAN_AVAILABLE and n_samples >= min_cluster_size:
            logger.info("Using HDBSCAN for clustering...")
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='cosine'
                )
                cluster_labels = clusterer.fit_predict(embeddings)
                
                # Count clusters
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                logger.info(f"HDBSCAN found {n_clusters} clusters")
                
                return cluster_labels
                
            except Exception as e:
                logger.warning(f"HDBSCAN failed: {e}, falling back to KMeans")
        
        # Fallback to KMeans
        logger.info("Using KMeans for clustering...")
        
        # Determine number of clusters
        n_clusters = min(max_clusters, max(min_clusters, n_samples // 5))
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        cluster_labels = kmeans.fit_predict(embeddings)
        self.cluster_centers = kmeans.cluster_centers_
        
        logger.info(f"KMeans found {n_clusters} clusters")
        
        return cluster_labels
    
    def find_cluster_representatives(self, emails: List[Dict], embeddings: np.ndarray, 
                                   cluster_labels: np.ndarray) -> Dict[int, int]:
        """Find representative emails for each cluster."""
        cluster_representatives = {}
        
        # Group emails by cluster
        clusters = {}
        for i, (email, embedding, label) in enumerate(zip(emails, embeddings, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((i, email, embedding))
        
        # Find representative for each cluster
        for cluster_id, cluster_items in clusters.items():
            if cluster_id == -1:  # Noise cluster in HDBSCAN
                continue
            
            if len(cluster_items) == 1:
                # Single item cluster
                cluster_representatives[cluster_id] = cluster_items[0][0]
            else:
                # Find item closest to cluster center
                cluster_embeddings = np.array([item[2] for item in cluster_items])
                
                if self.cluster_centers is not None and cluster_id < len(self.cluster_centers):
                    # Use KMeans cluster center
                    center = self.cluster_centers[cluster_id]
                else:
                    # Use mean of cluster embeddings
                    center = np.mean(cluster_embeddings, axis=0)
                
                # Calculate distances to center
                distances = np.linalg.norm(cluster_embeddings - center, axis=1)
                representative_idx = np.argmin(distances)
                
                cluster_representatives[cluster_id] = cluster_items[representative_idx][0]
        
        logger.info(f"Found representatives for {len(cluster_representatives)} clusters")
        
        return cluster_representatives
    
    def cluster_emails(self, emails: List[Dict], min_cluster_size: int = 2) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """Cluster emails and return representatives and cluster mappings."""
        if not emails:
            return {}, {}
        
        logger.info(f"Clustering {len(emails)} emails...")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(emails)
        self.embeddings = embeddings
        
        # Cluster embeddings
        cluster_labels = self.cluster_embeddings(embeddings, min_cluster_size)
        self.cluster_labels = cluster_labels
        
        # Find representatives
        representatives = self.find_cluster_representatives(emails, embeddings, cluster_labels)
        
        # Create cluster mappings
        cluster_mappings = {}
        for i, label in enumerate(cluster_labels):
            if label not in cluster_mappings:
                cluster_mappings[label] = []
            cluster_mappings[label].append(i)
        
        return representatives, cluster_mappings
    
    def get_cluster_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix between all email pairs."""
        return cosine_similarity(embeddings)
    
    def find_duplicates(self, emails: List[Dict], similarity_threshold: float = 0.95) -> List[List[int]]:
        """Find near-duplicate emails based on similarity threshold."""
        if not emails:
            return []
        
        embeddings = self.generate_embeddings(emails)
        similarity_matrix = self.get_cluster_similarity_matrix(embeddings)
        
        duplicates = []
        processed = set()
        
        for i in range(len(emails)):
            if i in processed:
                continue
            
            duplicate_group = [i]
            for j in range(i + 1, len(emails)):
                if j in processed:
                    continue
                
                if similarity_matrix[i, j] >= similarity_threshold:
                    duplicate_group.append(j)
                    processed.add(j)
            
            if len(duplicate_group) > 1:
                duplicates.append(duplicate_group)
                processed.update(duplicate_group)
        
        logger.info(f"Found {len(duplicates)} duplicate groups")
        
        return duplicates
    
    def save_clustering_results(self, output_file: Path, emails: List[Dict], 
                              representatives: Dict[int, int], cluster_mappings: Dict[int, List[int]]):
        """Save clustering results to file."""
        results = {
            'representatives': representatives,
            'cluster_mappings': cluster_mappings,
            'cluster_summary': {}
        }
        
        # Add cluster summaries
        for cluster_id, email_indices in cluster_mappings.items():
            if cluster_id == -1:  # Skip noise cluster
                continue
            
            cluster_emails = [emails[i] for i in email_indices]
            representative_idx = representatives.get(cluster_id)
            
            results['cluster_summary'][cluster_id] = {
                'size': len(email_indices),
                'representative_fingerprint': cluster_emails[representative_idx].get('fingerprint') if representative_idx is not None else None,
                'representative_subject': cluster_emails[representative_idx].get('subject') if representative_idx is not None else None
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Clustering results saved to {output_file}")


def main():
    """CLI interface for email clustering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Email clustering and deduplication")
    parser.add_argument("input_file", help="Input JSONL file with emails")
    parser.add_argument("output_file", help="Output JSON file with clustering results")
    parser.add_argument("--min-cluster-size", type=int, default=2, help="Minimum cluster size")
    parser.add_argument("--find-duplicates", action="store_true", help="Find near-duplicates")
    parser.add_argument("--similarity-threshold", type=float, default=0.95, help="Similarity threshold for duplicates")
    
    args = parser.parse_args()
    
    # Load emails
    emails = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                emails.append(json.loads(line))
    
    clusterer = EmailClusterer()
    
    if args.find_duplicates:
        # Find duplicates
        duplicates = clusterer.find_duplicates(emails, args.similarity_threshold)
        
        # Save duplicates
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(duplicates, f, indent=2)
        
        print(f"Found {len(duplicates)} duplicate groups")
        
    else:
        # Cluster emails
        representatives, cluster_mappings = clusterer.cluster_emails(emails, args.min_cluster_size)
        
        # Save results
        clusterer.save_clustering_results(Path(args.output_file), emails, representatives, cluster_mappings)
        
        print(f"Clustering complete: {len(representatives)} clusters found")


if __name__ == "__main__":
    main()
