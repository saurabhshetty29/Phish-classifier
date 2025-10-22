"""
LLM batch processing module for final email classification.

This module handles batching representative emails and sending them to an LLM
for final classification, with robust JSON parsing and retry logic.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import openai
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMBatchClassifier:
    """Handles batch LLM classification of representative emails."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # System prompt for email classification
        self.system_prompt = """You are an expert email security analyst. Your task is to classify emails as either 'good' (legitimate) or 'bad' (phishing/spam).

For each email, analyze:
1. Sender domain and display name consistency
2. Subject line urgency and suspicious language
3. Content for phishing indicators (urgent action, links, requests for personal info)
4. URL domains and redirect services
5. Overall email structure and authenticity

Return ONLY a valid JSON array with this exact format:
[{"id": "email_id", "label": "good" or "bad", "confidence": 0.0-1.0, "reason": "brief explanation"}]

Be conservative - when in doubt, classify as 'bad'. Focus on security threats."""
    
    def format_email_for_llm(self, email_data: Dict, email_id: str) -> str:
        """Format email data for LLM input."""
        from_header = email_data.get('from', '')
        subject = email_data.get('subject', '')
        snippet = email_data.get('snippet', '')
        url_domains = email_data.get('url_domains', [])
        sender_domain = email_data.get('sender_domain', '')
        
        # Truncate snippet if too long
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        
        formatted = f"id:{email_id} | from:'{from_header}' | subject:'{subject}' | snippet:'{snippet}'"
        
        if url_domains:
            formatted += f" | url_domains:{url_domains[:5]}"  # Limit to first 5 URLs
        
        if sender_domain:
            formatted += f" | sender_domain:'{sender_domain}'"
        
        return formatted
    
    def create_batch_prompt(self, emails: List[Dict], email_ids: List[str]) -> str:
        """Create a batch prompt for multiple emails."""
        if not emails:
            return ""
        
        # Format each email
        formatted_emails = []
        for i, (email, email_id) in enumerate(zip(emails, email_ids), 1):
            formatted = self.format_email_for_llm(email, email_id)
            formatted_emails.append(f"{i}) {formatted}")
        
        # Combine into batch prompt
        batch_prompt = "Classify these emails:\n\n" + "\n\n".join(formatted_emails)
        
        return batch_prompt
    
    def parse_llm_response(self, response: str, expected_ids: List[str]) -> List[Dict]:
        """Parse LLM response and validate JSON format."""
        try:
            # Clean response - remove any markdown formatting
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            results = json.loads(response)
            
            if not isinstance(results, list):
                raise ValueError("Response must be a JSON array")
            
            # Validate and clean results
            validated_results = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                
                # Ensure required fields
                if 'id' not in result or 'label' not in result:
                    continue
                
                # Validate label
                label = result['label'].lower()
                if label not in ['good', 'bad']:
                    continue
                
                # Validate confidence
                confidence = result.get('confidence', 0.5)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.5
                confidence = max(0.0, min(1.0, float(confidence)))
                
                # Ensure reason exists
                reason = result.get('reason', 'No reason provided')
                
                validated_result = {
                    'id': str(result['id']),
                    'label': label.upper(),
                    'confidence': confidence,
                    'reason': str(reason)
                }
                
                validated_results.append(validated_result)
            
            # Check if we got results for all expected IDs
            result_ids = {r['id'] for r in validated_results}
            missing_ids = set(expected_ids) - result_ids
            
            if missing_ids:
                logger.warning(f"Missing results for IDs: {missing_ids}")
            
            return validated_results
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return []
    
    def call_llm(self, prompt: str) -> str:
        """Make API call to LLM."""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please provide API key.")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def classify_batch(self, emails: List[Dict], email_ids: List[str]) -> List[Dict]:
        """Classify a batch of emails using LLM."""
        if not emails:
            return []
        
        if len(emails) != len(email_ids):
            raise ValueError("Number of emails must match number of email IDs")
        
        # Create batch prompt
        prompt = self.create_batch_prompt(emails, email_ids)
        
        # Make API call with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"LLM batch classification attempt {attempt + 1}/{self.max_retries}")
                response = self.call_llm(prompt)
                
                # Parse response
                results = self.parse_llm_response(response, email_ids)
                
                if results:
                    logger.info(f"Successfully classified {len(results)} emails")
                    return results
                else:
                    logger.warning(f"Empty results on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error("All retry attempts failed")
                    raise
        
        return []
    
    def classify_representatives(self, representative_emails: List[Dict], 
                               batch_size: int = 10) -> List[Dict]:
        """Classify representative emails in batches."""
        if not representative_emails:
            return []
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(representative_emails), batch_size):
            batch_emails = representative_emails[i:i + batch_size]
            batch_ids = [f"rep_{i + j}" for j in range(len(batch_emails))]
            
            logger.info(f"Processing batch {i // batch_size + 1}: {len(batch_emails)} emails")
            
            try:
                batch_results = self.classify_batch(batch_emails, batch_ids)
                all_results.extend(batch_results)
                
                # Add small delay between batches to respect rate limits
                if i + batch_size < len(representative_emails):
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i // batch_size + 1}: {e}")
                # Continue with next batch
                continue
        
        logger.info(f"LLM classification complete: {len(all_results)} results")
        return all_results
    
    def propagate_labels(self, llm_results: List[Dict], cluster_mappings: Dict[int, List[int]], 
                        representative_mapping: Dict[int, int]) -> List[Dict]:
        """Propagate LLM labels to all emails in clusters."""
        propagated_results = []
        
        # Create mapping from representative ID to LLM result
        llm_result_map = {result['id']: result for result in llm_results}
        
        for cluster_id, email_indices in cluster_mappings.items():
            if cluster_id == -1:  # Skip noise cluster
                continue
            
            # Get representative result
            rep_idx = representative_mapping.get(cluster_id)
            if rep_idx is None:
                continue
            
            rep_id = f"rep_{rep_idx}"
            rep_result = llm_result_map.get(rep_id)
            
            if not rep_result:
                continue
            
            # Propagate to all emails in cluster
            for email_idx in email_indices:
                propagated_result = {
                    'email_index': email_idx,
                    'label': rep_result['label'],
                    'confidence': rep_result['confidence'] * 0.9,  # Slightly reduce confidence for propagated labels
                    'reason': f"Propagated from cluster {cluster_id}: {rep_result['reason']}",
                    'source': 'llm_propagated'
                }
                propagated_results.append(propagated_result)
        
        logger.info(f"Propagated labels to {len(propagated_results)} emails")
        return propagated_results


def main():
    """CLI interface for LLM batch classification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM batch email classification")
    parser.add_argument("emails_file", help="JSONL file with emails")
    parser.add_argument("representatives_file", help="JSON file with cluster representatives")
    parser.add_argument("output_file", help="Output JSONL file with classifications")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model to use")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for LLM calls")
    
    args = parser.parse_args()
    
    # Load emails
    emails = []
    with open(args.emails_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                emails.append(json.loads(line))
    
    # Load representatives
    with open(args.representatives_file, 'r', encoding='utf-8') as f:
        rep_data = json.load(f)
    
    representatives = rep_data['representatives']
    cluster_mappings = rep_data['cluster_mappings']
    
    # Initialize classifier
    classifier = LLMBatchClassifier(api_key=args.api_key, model=args.model)
    
    # Get representative emails
    rep_emails = [emails[rep_data['representatives'][cluster_id]] 
                  for cluster_id in rep_data['representatives']]
    
    # Classify representatives
    llm_results = classifier.classify_representatives(rep_emails, args.batch_size)
    
    # Propagate labels
    propagated_results = classifier.propagate_labels(llm_results, cluster_mappings, representatives)
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in propagated_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"LLM classification complete: {len(propagated_results)} results saved")


if __name__ == "__main__":
    main()
