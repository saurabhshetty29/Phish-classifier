# Phish Classifier

A complete, modular Python project for classifying `.eml` emails as GOOD or BAD (phishing/spam) using a tiered hybrid pipeline optimized to minimize expensive LLM calls.

## 🏗️ Architecture Overview

The system uses a 6-stage pipeline that progressively filters emails through increasingly sophisticated (and expensive) classification methods:

```
📧 .eml Files → 🔍 Parse → 📋 Rules → 🤖 ML → 🎯 Cluster → 🧠 LLM → 📊 Final Labels
```

### Pipeline Stages

1. **Email Parsing** (`parse_eml.py`) - Extract structured data from .eml files
2. **Rule-Based Filtering** (`rules.py`) - Zero-cost heuristics for obvious good/bad emails
3. **ML Classification** (`model_tfidf_lr.py`) - Lightweight TF-IDF + Logistic Regression
4. **Clustering** (`dedupe_cluster.py`) - Group similar emails using embeddings
5. **LLM Classification** (`llm_batch.py`) - Batch process representative emails
6. **Caching** (`cache_store.py`) - SQLite-based result caching and model training data

## 📂 Project Structure

```
phish_classifier/
├── data/
│   └── raw_eml/               # Input .eml files
├── outputs/
│   ├── parsed_emails.jsonl    # Parsed structured data
│   ├── labels.jsonl           # Final labels (good/bad)
│   └── clustering_results.json # Cluster mappings
├── models/                    # Trained ML models
├── cache/                     # SQLite cache database
├── src/
│   ├── parse_eml.py           # Step 1: Email parsing
│   ├── rules.py               # Step 2: Rule-based filters
│   ├── model_tfidf_lr.py      # Step 3: ML classifier
│   ├── dedupe_cluster.py      # Step 4: Embeddings + clustering
│   ├── llm_batch.py           # Step 5: LLM batch processing
│   ├── cache_store.py         # SQLite caching system
│   └── pipeline.py            # Main orchestrator
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Installation

1. **Clone and setup:**
```bash
cd phish_classifier
pip install -r requirements.txt
```

2. **Prepare your data:**
```bash
# Copy .eml files to data/raw_eml/
cp /path/to/your/emails/*.eml data/raw_eml/
```

3. **Run the complete pipeline:**
```bash
# Basic run (no LLM)
python src/pipeline.py --input-dir data/raw_eml --output-dir outputs

# With LLM classification (requires OpenAI API key)
python src/pipeline.py --input-dir data/raw_eml --output-dir outputs --openai-api-key YOUR_API_KEY

# Train ML model first
python src/pipeline.py --train --input-dir data/raw_eml --output-dir outputs
```

### Individual Components

Each module can be run independently:

```bash
# Parse emails only
python src/parse_eml.py data/raw_eml outputs/parsed_emails.jsonl

# Rule-based classification
python src/rules.py outputs/parsed_emails.jsonl outputs/rule_labels.jsonl

# Train ML model
python src/model_tfidf_lr.py --train --emails outputs/parsed_emails.jsonl --labels outputs/rule_labels.jsonl

# Cluster emails
python src/dedupe_cluster.py outputs/parsed_emails.jsonl outputs/clusters.json

# LLM classification
python src/llm_batch.py outputs/parsed_emails.jsonl outputs/clusters.json outputs/llm_labels.jsonl --api-key YOUR_API_KEY
```

## 🔧 Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"  # For LLM classification
```

### Pipeline Options

```bash
python src/pipeline.py --help

# Available options:
--input-dir DIR          # Directory containing .eml files
--output-dir DIR         # Output directory for results
--model-dir DIR          # Directory for ML models
--cache-dir DIR          # Directory for SQLite cache
--train                  # Train ML model before classification
--no-llm                 # Skip LLM classification step
--openai-api-key KEY     # OpenAI API key for LLM calls
--step STEP              # Run specific pipeline step (1-6, all)
--clear-cache            # Clear old cache entries
--cache-stats            # Show cache statistics
```

## 📊 Pipeline Details

### Stage 1: Email Parsing
- Extracts headers, body content, URLs, authentication results
- Generates unique fingerprints for caching
- Handles HTML cleaning and encoding issues
- Output: `parsed_emails.jsonl`

### Stage 2: Rule-Based Classification
- **Zero-cost heuristics** for obvious classifications
- Domain mismatch detection
- Phishing keyword analysis
- Suspicious URL pattern matching
- Authentication failure detection
- Output: GOOD/BAD/UNKNOWN labels

### Stage 3: ML Classification
- **TF-IDF + Logistic Regression** for high-confidence predictions
- Features: subject, body, URL count, auth flags, text patterns
- Confidence thresholds: ≥95% auto-label, ≤5% ambiguous
- Output: Auto-labeled emails + ambiguous subset

### Stage 4: Clustering
- **Sentence transformers** (all-MiniLM-L6-v2) for embeddings
- HDBSCAN clustering (KMeans fallback)
- Groups similar phishing campaigns
- Output: Representative emails per cluster

### Stage 5: LLM Classification
- **Batch processing** of cluster representatives
- JSON-only prompts for structured responses
- Robust parsing with retry logic
- Label propagation to cluster members
- Output: Final classifications for ambiguous emails

### Stage 6: Caching & Final Labels
- **SQLite-based caching** for all results
- Combines all sources with confidence weighting
- Generates final labeled dataset
- Supports model retraining over time

## 🎯 Key Features

### Cost Optimization
- **Rule-based filtering** eliminates 60-80% of emails without ML/LLM costs
- **Clustering** reduces LLM calls by 80-90% through deduplication
- **Caching** prevents reprocessing of identical emails
- **Batch processing** minimizes API call overhead

### Accuracy & Reliability
- **Multi-stage validation** with confidence scoring
- **Source tracking** for audit trails
- **Robust error handling** and retry logic
- **Model retraining** support for continuous improvement

### Scalability
- **Modular design** allows independent component updates
- **SQLite caching** supports large datasets
- **Batch processing** handles thousands of emails efficiently
- **Memory-efficient** streaming for large files

## 📈 Performance Metrics

The pipeline is designed to achieve:
- **90%+ accuracy** on phishing detection
- **80%+ reduction** in LLM API costs through clustering
- **Sub-second processing** per email for rules/ML stages
- **Scalable to 100K+ emails** with proper caching

## 🔍 Example Output

```json
{
  "fingerprint": "a1b2c3d4e5f6...",
  "label": "BAD",
  "confidence": 0.95,
  "source": "rules",
  "context": {
    "risk_score": 4,
    "risk_factors": {
      "domain_mismatch": true,
      "phishing_keywords": true,
      "suspicious_urls": false,
      "auth_failure": true
    }
  },
  "all_sources": ["rules"]
}
```

## 🛠️ Extending the System

### Adding New Rules
Edit `src/rules.py` to add custom heuristics:
```python
def check_custom_pattern(self, email_data: Dict) -> bool:
    # Your custom logic here
    return suspicious_condition
```

### Custom ML Features
Modify `src/model_tfidf_lr.py` to add new features:
```python
def extract_custom_features(self, email_data: Dict) -> Dict[str, float]:
    # Add your features here
    return features
```

### Different LLM Providers
Extend `src/llm_batch.py` to support other LLM APIs:
```python
def call_custom_llm(self, prompt: str) -> str:
    # Your LLM integration here
    return response
```

## 🐛 Troubleshooting

### Common Issues

1. **No emails parsed**: Check .eml file format and encoding
2. **ML model errors**: Run with `--train` flag first
3. **LLM API errors**: Verify API key and rate limits
4. **Memory issues**: Process emails in smaller batches

### Debug Mode

```bash
# Enable verbose logging
export PYTHONPATH=src
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from pipeline import PhishingClassificationPipeline
pipeline = PhishingClassificationPipeline()
pipeline.run_full_pipeline()
"
```
### Dashboard View
<img width="1416" height="451" alt="screen1" src="https://github.com/user-attachments/assets/c16d114d-af74-4fd0-81a7-1400cb9be45e" />
<img width="1352" height="730" alt="Screen-2" src="https://github.com/user-attachments/assets/6544dbcd-3f0a-4168-a75d-2e25ea28c0a7" />
<img width="1371" height="796" alt="Screen-3" src="https://github.com/user-attachments/assets/cd7fbaab-73e1-4447-8c39-8ad5933ce2b4" />
