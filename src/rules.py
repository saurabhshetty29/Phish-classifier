"""
Rule-based filtering module for phishing classification pipeline.

This module implements heuristics to automatically label emails as GOOD, BAD, or UNKNOWN
based on domain analysis, keyword detection, URL patterns, and authentication results.
"""

import logging
import re
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RuleBasedClassifier:
    """Rule-based classifier using heuristics to identify phishing emails."""
    
    def __init__(self):
        # Known phishing keywords
        self.phishing_keywords = [
            "verify your account", "urgent action", "reset password", 
            "wire transfer", "invoice", "gift card", "docusign",
            "suspended", "expired", "immediate action", "click here",
            "verify now", "account locked", "security alert",
            "payment failed", "billing issue", "account verification",
            "suspicious activity", "unusual login", "confirm identity",
            "update payment", "renew subscription", "expiring soon"
        ]
        
        # Suspicious URL shorteners and redirect services
        self.suspicious_domains = [
            "bit.ly", "tinyurl.com", "t.co", "goo.gl", "short.link",
            "is.gd", "v.gd", "ow.ly", "buff.ly", "rebrand.ly",
            "tiny.cc", "shorturl.at", "cutt.ly", "short.link",
            "rb.gy", "bit.do", "short.link", "tiny.one"
        ]
        
        # Trusted domains (major email providers, banks, etc.)
        self.trusted_domains = [
            "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
            "apple.com", "microsoft.com", "google.com", "amazon.com",
            "paypal.com", "ebay.com", "linkedin.com", "facebook.com",
            "twitter.com", "instagram.com", "netflix.com", "spotify.com",
            "dropbox.com", "adobe.com", "salesforce.com", "slack.com"
        ]
        
        # Compile regex patterns for efficiency
        self.phishing_patterns = [re.compile(keyword, re.IGNORECASE) for keyword in self.phishing_keywords]
        self.suspicious_domain_patterns = [re.compile(domain, re.IGNORECASE) for domain in self.suspicious_domains]
        self.trusted_domain_patterns = [re.compile(domain, re.IGNORECASE) for domain in self.trusted_domains]
    
    def check_domain_mismatch(self, email_data: Dict) -> bool:
        """Check if display name doesn't match sender domain."""
        from_header = email_data.get('from', '')
        sender_domain = email_data.get('sender_domain', '')
        
        if not from_header or not sender_domain:
            return False
        
        # Extract display name (text before <email>)
        display_name_match = re.match(r'^([^<]+)<', from_header)
        if not display_name_match:
            return False
        
        display_name = display_name_match.group(1).strip().lower()
        
        # Check if display name contains a domain that doesn't match sender domain
        domain_in_display = re.search(r'@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', display_name)
        if domain_in_display:
            display_domain = domain_in_display.group(1).lower()
            return display_domain != sender_domain
        
        # Check for common company names that might indicate spoofing
        suspicious_companies = ['paypal', 'apple', 'microsoft', 'google', 'amazon', 'netflix']
        for company in suspicious_companies:
            if company in display_name and company not in sender_domain:
                return True
        
        return False
    
    def check_phishing_keywords(self, email_data: Dict) -> Tuple[bool, List[str]]:
        """Check for phishing keywords in subject and body."""
        subject = email_data.get('subject', '').lower()
        snippet = email_data.get('snippet', '').lower()
        text_to_check = f"{subject} {snippet}"
        
        found_keywords = []
        for pattern in self.phishing_patterns:
            if pattern.search(text_to_check):
                found_keywords.append(pattern.pattern)
        
        return len(found_keywords) >= 2, found_keywords
    
    def check_suspicious_urls(self, email_data: Dict) -> Tuple[bool, List[str]]:
        """Check for suspicious URL domains."""
        url_domains = email_data.get('url_domains', [])
        suspicious_found = []
        
        for domain in url_domains:
            for pattern in self.suspicious_domain_patterns:
                if pattern.search(domain):
                    suspicious_found.append(domain)
        
        return len(suspicious_found) > 0, suspicious_found
    
    def check_authentication_failures(self, email_data: Dict) -> bool:
        """Check if email authentication (SPF/DKIM/DMARC) failed."""
        spf_pass = email_data.get('spf_pass', False)
        dkim_pass = email_data.get('dkim_pass', False)
        dmarc_pass = email_data.get('dmarc_pass', False)
        
        # If any authentication method failed, it's suspicious
        return not (spf_pass or dkim_pass or dmarc_pass)
    
    def check_trusted_domain(self, email_data: Dict) -> bool:
        """Check if email is from a trusted domain."""
        sender_domain = email_data.get('sender_domain', '')
        
        if not sender_domain:
            return False
        
        for pattern in self.trusted_domain_patterns:
            if pattern.search(sender_domain):
                return True
        
        return False
    
    def check_urgent_language(self, email_data: Dict) -> bool:
        """Check for urgent language patterns."""
        subject = email_data.get('subject', '').lower()
        snippet = email_data.get('snippet', '').lower()
        text_to_check = f"{subject} {snippet}"
        
        urgent_patterns = [
            r'\b(urgent|asap|immediately|right now|expires?|deadline)\b',
            r'\b(act now|don\'t wait|limited time|hurry)\b',
            r'\b(click here|verify now|confirm immediately)\b',
            r'\b(account will be|suspended|locked|closed)\b'
        ]
        
        for pattern in urgent_patterns:
            if re.search(pattern, text_to_check):
                return True
        
        return False
    
    def check_suspicious_headers(self, email_data: Dict) -> bool:
        """Check for suspicious email headers."""
        from_header = email_data.get('from', '')
        
        # Check for suspicious patterns in From header
        suspicious_patterns = [
            r'<[^>]*@[^>]*>',  # Multiple email addresses
            r'[^\x00-\x7F]',   # Non-ASCII characters
            r'\b(no-reply|noreply|donotreply)\b',  # No-reply addresses
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, from_header, re.IGNORECASE):
                return True
        
        return False
    
    def calculate_risk_score(self, email_data: Dict) -> Tuple[int, Dict[str, bool]]:
        """Calculate risk score based on various heuristics."""
        risk_factors = {}
        score = 0
        
        # Domain mismatch (high risk)
        domain_mismatch = self.check_domain_mismatch(email_data)
        risk_factors['domain_mismatch'] = domain_mismatch
        if domain_mismatch:
            score += 3
        
        # Phishing keywords (medium-high risk)
        has_keywords, found_keywords = self.check_phishing_keywords(email_data)
        risk_factors['phishing_keywords'] = has_keywords
        if has_keywords:
            score += 2
        
        # Suspicious URLs (medium risk)
        has_suspicious_urls, suspicious_urls = self.check_suspicious_urls(email_data)
        risk_factors['suspicious_urls'] = has_suspicious_urls
        if has_suspicious_urls:
            score += 2
        
        # Authentication failures (medium risk)
        auth_failure = self.check_authentication_failures(email_data)
        risk_factors['auth_failure'] = auth_failure
        if auth_failure:
            score += 2
        
        # Urgent language (low-medium risk)
        urgent_language = self.check_urgent_language(email_data)
        risk_factors['urgent_language'] = urgent_language
        if urgent_language:
            score += 1
        
        # Suspicious headers (low risk)
        suspicious_headers = self.check_suspicious_headers(email_data)
        risk_factors['suspicious_headers'] = suspicious_headers
        if suspicious_headers:
            score += 1
        
        return score, risk_factors
    
    def classify_email(self, email_data: Dict) -> Tuple[str, float, Dict]:
        """Classify email as GOOD, BAD, or UNKNOWN based on rules."""
        risk_score, risk_factors = self.calculate_risk_score(email_data)
        
        # Check if it's from a trusted domain and clean
        is_trusted = self.check_trusted_domain(email_data)
        is_clean = risk_score <= 1
        
        # Strong indicators for BAD
        strong_bad_indicators = [
            risk_factors['domain_mismatch'],
            risk_factors['phishing_keywords'],
            risk_factors['suspicious_urls'],
            risk_factors['auth_failure']
        ]
        
        strong_bad_count = sum(strong_bad_indicators)
        
        # Classification logic
        if strong_bad_count >= 2 or risk_score >= 5:
            label = "BAD"
            confidence = min(0.95, 0.6 + (risk_score * 0.1))
        elif is_trusted and is_clean:
            label = "GOOD"
            confidence = 0.9
        else:
            label = "UNKNOWN"
            confidence = 0.5
        
        # Additional context
        context = {
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'is_trusted_domain': is_trusted,
            'strong_bad_indicators': strong_bad_count
        }
        
        return label, confidence, context
    
    def classify_batch(self, emails: List[Dict]) -> List[Dict]:
        """Classify a batch of emails using rule-based heuristics."""
        results = []
        
        for email_data in emails:
            try:
                label, confidence, context = self.classify_email(email_data)
                
                result = {
                    'fingerprint': email_data.get('fingerprint'),
                    'label': label,
                    'confidence': confidence,
                    'source': 'rules',
                    'context': context
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to classify email {email_data.get('fingerprint', 'unknown')}: {e}")
                # Default to UNKNOWN for failed classifications
                result = {
                    'fingerprint': email_data.get('fingerprint'),
                    'label': 'UNKNOWN',
                    'confidence': 0.0,
                    'source': 'rules',
                    'context': {'error': str(e)}
                }
                results.append(result)
        
        return results


def main():
    """CLI interface for rule-based classification."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Rule-based email classification")
    parser.add_argument("input_file", help="Input JSONL file with parsed emails")
    parser.add_argument("output_file", help="Output JSONL file with classifications")
    
    args = parser.parse_args()
    
    classifier = RuleBasedClassifier()
    
    # Read emails
    emails = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                emails.append(json.loads(line))
    
    # Classify emails
    results = classifier.classify_batch(emails)
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Print summary
    label_counts = {}
    for result in results:
        label = result['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Classification complete:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
