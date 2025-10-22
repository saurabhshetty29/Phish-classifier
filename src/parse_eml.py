"""
Email parsing module for phishing classification pipeline.

This module handles parsing of .eml files, extracting structured data including
headers, body content, URLs, and authentication results.
"""

import email
import hashlib
import json
import logging
import re
from email.header import decode_header
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import tqdm
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailParser:
    """Parser for .eml files that extracts structured email data."""
    
    def __init__(self):
        self.phishing_keywords = [
            "verify your account", "urgent action", "reset password", 
            "wire transfer", "invoice", "gift card", "DocuSign",
            "suspended", "expired", "immediate action", "click here",
            "verify now", "account locked", "security alert"
        ]
        
        self.suspicious_domains = [
            "bit.ly", "tinyurl.com", "t.co", "goo.gl", "short.link",
            "is.gd", "v.gd", "ow.ly", "buff.ly", "rebrand.ly"
        ]
    
    def decode_header_value(self, value: str) -> str:
        """Decode email header value handling encoding issues."""
        if not value:
            return ""
        
        try:
            decoded_parts = decode_header(value)
            decoded_string = ""
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    if encoding:
                        decoded_string += part.decode(encoding, errors='ignore')
                    else:
                        decoded_string += part.decode('utf-8', errors='ignore')
                else:
                    decoded_string += str(part)
            return decoded_string.strip()
        except Exception as e:
            logger.warning(f"Failed to decode header value: {e}")
            return str(value)
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text content."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text, re.IGNORECASE)
        return list(set(urls))  # Remove duplicates
    
    def extract_domains_from_urls(self, urls: List[str]) -> List[str]:
        """Extract domains from URLs."""
        domains = []
        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                if domain:
                    domains.append(domain)
            except Exception as e:
                logger.warning(f"Failed to parse URL {url}: {e}")
        return list(set(domains))
    
    def clean_html_body(self, html_content: str) -> str:
        """Clean HTML content and extract text."""
        if not html_content:
            return ""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.warning(f"Failed to clean HTML: {e}")
            return html_content
    
    def extract_authentication_flags(self, headers: Dict[str, str]) -> Dict[str, bool]:
        """Extract SPF, DKIM, DMARC authentication results."""
        auth_results = {
            'spf_pass': False,
            'dkim_pass': False,
            'dmarc_pass': False
        }
        
        auth_header = headers.get('Authentication-Results', '').lower()
        if not auth_header:
            return auth_results
        
        # Check SPF
        if 'spf=pass' in auth_header:
            auth_results['spf_pass'] = True
        
        # Check DKIM
        if 'dkim=pass' in auth_header:
            auth_results['dkim_pass'] = True
        
        # Check DMARC
        if 'dmarc=pass' in auth_header:
            auth_results['dmarc_pass'] = True
        
        return auth_results
    
    def extract_sender_domain(self, from_header: str) -> str:
        """Extract domain from From header."""
        if not from_header:
            return ""
        
        # Look for email address in angle brackets
        email_match = re.search(r'<([^>]+@[^>]+)>', from_header)
        if email_match:
            email_addr = email_match.group(1)
        else:
            # Look for email address without brackets
            email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', from_header)
            if email_match:
                email_addr = email_match.group(1)
            else:
                return ""
        
        # Extract domain
        domain_match = re.search(r'@(.+)', email_addr)
        if domain_match:
            return domain_match.group(1).lower()
        
        return ""
    
    def generate_fingerprint(self, email_data: Dict) -> str:
        """Generate a unique fingerprint for the email."""
        # Create a string representation of key fields for hashing
        key_fields = [
            email_data.get('from', ''),
            email_data.get('subject', ''),
            email_data.get('snippet', '')[:200],  # First 200 chars of snippet
            str(sorted(email_data.get('url_domains', [])))
        ]
        
        content = '|'.join(key_fields)
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def parse_email_file(self, file_path: Path) -> Optional[Dict]:
        """Parse a single .eml file and return structured data."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                email_content = f.read()
            
            # Parse email
            msg = email.message_from_string(email_content)
            
            # Extract headers
            headers = {}
            for header in ['From', 'To', 'Subject', 'Date', 'Message-ID', 'Authentication-Results']:
                value = msg.get(header, '')
                headers[header] = self.decode_header_value(value)
            
            # Extract body content
            body = ""
            html_body = ""
            
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    elif content_type == "text/html":
                        html_body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                content_type = msg.get_content_type()
                if content_type == "text/plain":
                    body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                elif content_type == "text/html":
                    html_body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            # Clean HTML body
            if html_body:
                body = self.clean_html_body(html_body)
            
            # Extract URLs and domains
            urls = self.extract_urls(body)
            url_domains = self.extract_domains_from_urls(urls)
            
            # Extract authentication flags
            auth_flags = self.extract_authentication_flags(headers)
            
            # Extract sender domain
            sender_domain = self.extract_sender_domain(headers.get('From', ''))
            
            # Create snippet (first 600 characters)
            snippet = body[:600] if body else ""
            
            # Generate fingerprint
            email_data = {
                'file_path': str(file_path),
                'from': headers.get('From', ''),
                'to': headers.get('To', ''),
                'subject': headers.get('Subject', ''),
                'date': headers.get('Date', ''),
                'message_id': headers.get('Message-ID', ''),
                'sender_domain': sender_domain,
                'snippet': snippet,
                'body': body,
                'urls': urls,
                'url_domains': url_domains,
                'spf_pass': auth_flags['spf_pass'],
                'dkim_pass': auth_flags['dkim_pass'],
                'dmarc_pass': auth_flags['dmarc_pass'],
                'url_count': len(urls),
                'has_html': bool(html_body),
                'content_length': len(body)
            }
            
            # Generate fingerprint
            email_data['fingerprint'] = self.generate_fingerprint(email_data)
            
            return email_data
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return None
    
    def parse_eml_directory(self, input_dir: Path, output_file: Path) -> int:
        """Parse all .eml files in a directory and save to JSONL."""
        input_path = Path(input_dir)
        output_path = Path(output_file)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Find all .eml files
        eml_files = list(input_path.glob("*.eml"))
        
        if not eml_files:
            logger.warning(f"No .eml files found in {input_dir}")
            return 0
        
        logger.info(f"Found {len(eml_files)} .eml files to parse")
        
        parsed_count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for eml_file in tqdm.tqdm(eml_files, desc="Parsing emails"):
                email_data = self.parse_email_file(eml_file)
                if email_data:
                    f.write(json.dumps(email_data, ensure_ascii=False) + '\n')
                    parsed_count += 1
        
        logger.info(f"Successfully parsed {parsed_count} emails")
        return parsed_count


def main():
    """CLI interface for email parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse .eml files for phishing classification")
    parser.add_argument("input_dir", help="Directory containing .eml files")
    parser.add_argument("output_file", help="Output JSONL file path")
    
    args = parser.parse_args()
    
    parser_instance = EmailParser()
    count = parser_instance.parse_eml_directory(Path(args.input_dir), Path(args.output_file))
    print(f"Parsed {count} emails successfully")


if __name__ == "__main__":
    main()
