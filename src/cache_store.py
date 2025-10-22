"""
SQLite-based caching system for email classification results.

This module provides persistent caching of classification results to avoid
reprocessing emails and to support model retraining over time.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationCache:
    """SQLite-based cache for email classification results."""
    
    def __init__(self, db_path: str = "cache/classifications.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create classifications table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classifications (
                    fingerprint TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create model_versions table for tracking model updates
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Create email_metadata table for storing email info
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_metadata (
                    fingerprint TEXT PRIMARY KEY,
                    file_path TEXT,
                    subject TEXT,
                    sender_domain TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_classifications_source 
                ON classifications(source)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_classifications_created 
                ON classifications(created_at)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_classifications_label 
                ON classifications(label)
            """)
            
            conn.commit()
    
    def get_classification(self, fingerprint: str) -> Optional[Dict]:
        """Get cached classification for an email fingerprint."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT label, confidence, source, context, created_at, updated_at
                FROM classifications
                WHERE fingerprint = ?
            """, (fingerprint,))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'fingerprint': fingerprint,
                    'label': result[0],
                    'confidence': result[1],
                    'source': result[2],
                    'context': json.loads(result[3]) if result[3] else {},
                    'created_at': result[4],
                    'updated_at': result[5]
                }
            
            return None
    
    def set_classification(self, fingerprint: str, label: str, confidence: float, 
                          source: str, context: Optional[Dict] = None) -> bool:
        """Store or update a classification result."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                context_json = json.dumps(context) if context else None
                
                cursor.execute("""
                    INSERT OR REPLACE INTO classifications 
                    (fingerprint, label, confidence, source, context, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (fingerprint, label, confidence, source, context_json))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store classification for {fingerprint}: {e}")
            return False
    
    def batch_set_classifications(self, classifications: List[Dict]) -> int:
        """Store multiple classifications in a batch."""
        success_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for classification in classifications:
                try:
                    fingerprint = classification['fingerprint']
                    label = classification['label']
                    confidence = classification['confidence']
                    source = classification['source']
                    context = classification.get('context', {})
                    
                    context_json = json.dumps(context) if context else None
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO classifications 
                        (fingerprint, label, confidence, source, context, updated_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (fingerprint, label, confidence, source, context_json))
                    
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to store classification: {e}")
                    continue
            
            conn.commit()
        
        logger.info(f"Stored {success_count}/{len(classifications)} classifications")
        return success_count
    
    def get_classifications_by_source(self, source: str) -> List[Dict]:
        """Get all classifications from a specific source."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT fingerprint, label, confidence, source, context, created_at, updated_at
                FROM classifications
                WHERE source = ?
                ORDER BY created_at DESC
            """, (source,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'fingerprint': row[0],
                    'label': row[1],
                    'confidence': row[2],
                    'source': row[3],
                    'context': json.loads(row[4]) if row[4] else {},
                    'created_at': row[5],
                    'updated_at': row[6]
                })
            
            return results
    
    def get_classifications_by_label(self, label: str) -> List[Dict]:
        """Get all classifications with a specific label."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT fingerprint, label, confidence, source, context, created_at, updated_at
                FROM classifications
                WHERE label = ?
                ORDER BY created_at DESC
            """, (label,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'fingerprint': row[0],
                    'label': row[1],
                    'confidence': row[2],
                    'source': row[3],
                    'context': json.loads(row[4]) if row[4] else {},
                    'created_at': row[5],
                    'updated_at': row[6]
                })
            
            return results
    
    def get_recent_classifications(self, hours: int = 24) -> List[Dict]:
        """Get classifications from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT fingerprint, label, confidence, source, context, created_at, updated_at
                FROM classifications
                WHERE created_at >= ?
                ORDER BY created_at DESC
            """, (cutoff_time.isoformat(),))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'fingerprint': row[0],
                    'label': row[1],
                    'confidence': row[2],
                    'source': row[3],
                    'context': json.loads(row[4]) if row[4] else {},
                    'created_at': row[5],
                    'updated_at': row[6]
                })
            
            return results
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total classifications
            cursor.execute("SELECT COUNT(*) FROM classifications")
            total_count = cursor.fetchone()[0]
            
            # Classifications by source
            cursor.execute("""
                SELECT source, COUNT(*) 
                FROM classifications 
                GROUP BY source
            """)
            by_source = dict(cursor.fetchall())
            
            # Classifications by label
            cursor.execute("""
                SELECT label, COUNT(*) 
                FROM classifications 
                GROUP BY label
            """)
            by_label = dict(cursor.fetchall())
            
            # Recent activity
            cursor.execute("""
                SELECT COUNT(*) 
                FROM classifications 
                WHERE created_at >= datetime('now', '-24 hours')
            """)
            recent_count = cursor.fetchone()[0]
            
            return {
                'total_classifications': total_count,
                'by_source': by_source,
                'by_label': by_label,
                'recent_24h': recent_count
            }
    
    def clear_old_classifications(self, days: int = 30) -> int:
        """Remove classifications older than specified days."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM classifications 
                WHERE created_at < ?
            """, (cutoff_time.isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
        
        logger.info(f"Cleared {deleted_count} old classifications")
        return deleted_count
    
    def store_email_metadata(self, email_data: Dict) -> bool:
        """Store email metadata for reference."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO email_metadata 
                    (fingerprint, file_path, subject, sender_domain)
                    VALUES (?, ?, ?, ?)
                """, (
                    email_data.get('fingerprint', ''),
                    email_data.get('file_path', ''),
                    email_data.get('subject', ''),
                    email_data.get('sender_domain', '')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store email metadata: {e}")
            return False
    
    def get_training_data(self, min_confidence: float = 0.7) -> Tuple[List[Dict], List[str]]:
        """Get high-confidence classifications for model training."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT c.fingerprint, c.label, c.confidence, c.source,
                       e.subject, e.sender_domain
                FROM classifications c
                LEFT JOIN email_metadata e ON c.fingerprint = e.fingerprint
                WHERE c.confidence >= ?
                ORDER BY c.confidence DESC
            """, (min_confidence,))
            
            emails = []
            labels = []
            
            for row in cursor.fetchall():
                # Reconstruct email data for training
                email_data = {
                    'fingerprint': row[0],
                    'subject': row[4] or '',
                    'sender_domain': row[5] or '',
                    'snippet': '',  # Would need to be stored separately
                    'body': '',      # Would need to be stored separately
                    'urls': [],
                    'url_domains': [],
                    'spf_pass': False,
                    'dkim_pass': False,
                    'dmarc_pass': False,
                    'url_count': 0,
                    'has_html': False,
                    'content_length': 0
                }
                
                emails.append(email_data)
                labels.append(row[1])
            
            return emails, labels
    
    def export_classifications(self, output_file: Path, source: Optional[str] = None) -> int:
        """Export classifications to JSONL file."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if source:
                cursor.execute("""
                    SELECT fingerprint, label, confidence, source, context, created_at, updated_at
                    FROM classifications
                    WHERE source = ?
                    ORDER BY created_at DESC
                """, (source,))
            else:
                cursor.execute("""
                    SELECT fingerprint, label, confidence, source, context, created_at, updated_at
                    FROM classifications
                    ORDER BY created_at DESC
                """)
            
            count = 0
            with open(output_file, 'w', encoding='utf-8') as f:
                for row in cursor.fetchall():
                    result = {
                        'fingerprint': row[0],
                        'label': row[1],
                        'confidence': row[2],
                        'source': row[3],
                        'context': json.loads(row[4]) if row[4] else {},
                        'created_at': row[5],
                        'updated_at': row[6]
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    count += 1
            
            logger.info(f"Exported {count} classifications to {output_file}")
            return count


def main():
    """CLI interface for cache management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Classification cache management")
    parser.add_argument("--db-path", default="cache/classifications.db", help="Database path")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--export", help="Export classifications to JSONL file")
    parser.add_argument("--clear-old", type=int, help="Clear classifications older than N days")
    parser.add_argument("--source", help="Filter by source for export")
    
    args = parser.parse_args()
    
    cache = ClassificationCache(args.db_path)
    
    if args.stats:
        stats = cache.get_cache_stats()
        print("Cache Statistics:")
        print(f"  Total classifications: {stats['total_classifications']}")
        print(f"  Recent (24h): {stats['recent_24h']}")
        print("  By source:")
        for source, count in stats['by_source'].items():
            print(f"    {source}: {count}")
        print("  By label:")
        for label, count in stats['by_label'].items():
            print(f"    {label}: {count}")
    
    elif args.export:
        count = cache.export_classifications(Path(args.export), args.source)
        print(f"Exported {count} classifications")
    
    elif args.clear_old:
        count = cache.clear_old_classifications(args.clear_old)
        print(f"Cleared {count} old classifications")


if __name__ == "__main__":
    main()
