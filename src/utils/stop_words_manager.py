#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stop Words Manager for RAG Multi-Application System

This module provides centralized stop words management with support for:
- Multiple languages (English, Spanish)
- Application-specific stop words
- NLTK integration
- Configurable settings
"""

import os
import yaml
from pathlib import Path
from typing import Set, List, Dict, Any, Optional
from loguru import logger

try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Install with: pip install nltk")


class StopWordsManager:
    """
    Centralized stop words management for the RAG system.
    
    Features:
    - Load stop words from YAML configuration
    - Support for multiple languages
    - Application-specific stop words
    - NLTK integration
    - Caching for performance
    """
    
    def __init__(self, config_path: str = "config/stop_words_config.yaml"):
        """
        Initialize the Stop Words Manager.
        
        Args:
            config_path: Path to the stop words configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._stop_words_cache: Dict[str, Set[str]] = {}
        self._nltk_initialized = False
        
        # Load configuration
        self._load_config()
        
        # Initialize NLTK if available and configured
        if self._should_use_nltk():
            self._initialize_nltk()
    
    def _load_config(self) -> None:
        """Load stop words configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.error(f"Stop words config file not found: {self.config_path}")
                self.config = self._get_default_config()
                return
            
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"Loaded stop words configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading stop words config: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is not available."""
        return {
            'stop_words': {
                'english': ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'],
                'spanish': ['que', 'de', 'la', 'el', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da']
            },
            'application_specific': {
                'technical': [],
                'sap': [],
                'darwin': []
            },
            'settings': {
                'min_word_length': 3,
                'max_key_terms': 10,
                'case_sensitive': False,
                'use_nltk_stopwords': True,
                'nltk_languages': ['english', 'spanish']
            }
        }
    
    def _should_use_nltk(self) -> bool:
        """Check if NLTK should be used based on configuration and availability."""
        return (NLTK_AVAILABLE and 
                self.config.get('settings', {}).get('use_nltk_stopwords', False))
    
    def _initialize_nltk(self) -> None:
        """Initialize NLTK and download required data."""
        if self._nltk_initialized:
            return
        
        try:
            # Download stopwords corpus if not already present
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                logger.info("Downloading NLTK stopwords corpus...")
                nltk.download('stopwords', quiet=True)
            
            self._nltk_initialized = True
            logger.info("NLTK stopwords initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLTK: {e}")
            self._nltk_initialized = False
    
    def get_stop_words(self, 
                      languages: Optional[List[str]] = None,
                      application: Optional[str] = None,
                      include_technical: bool = True) -> Set[str]:
        """
        Get comprehensive stop words set.
        
        Args:
            languages: List of languages to include (default: ['english', 'spanish'])
            application: Specific application to include stop words for
            include_technical: Whether to include technical stop words
            
        Returns:
            Set of stop words
        """
        if languages is None:
            languages = ['english', 'spanish']
        
        # Create cache key
        cache_key = f"{'-'.join(sorted(languages))}_{application}_{include_technical}"
        
        # Return cached result if available
        if cache_key in self._stop_words_cache:
            return self._stop_words_cache[cache_key]
        
        # Build stop words set
        stop_words = set()
        
        # Add language-specific stop words from config
        config_stop_words = self.config.get('stop_words', {})
        for lang in languages:
            if lang in config_stop_words:
                lang_words = config_stop_words[lang]
                if isinstance(lang_words, list):
                    stop_words.update(word.lower() for word in lang_words)
        
        # Add NLTK stop words if available and configured
        if self._should_use_nltk() and self._nltk_initialized:
            nltk_languages = self.config.get('settings', {}).get('nltk_languages', [])
            for lang in languages:
                if lang in nltk_languages:
                    try:
                        nltk_stop_words = set(stopwords.words(lang))
                        stop_words.update(word.lower() for word in nltk_stop_words)
                        logger.debug(f"Added {len(nltk_stop_words)} NLTK stop words for {lang}")
                    except Exception as e:
                        logger.warning(f"Could not load NLTK stop words for {lang}: {e}")
        
        # Add technical stop words if requested
        if include_technical:
            technical_words = self.config.get('application_specific', {}).get('technical', [])
            if isinstance(technical_words, list):
                stop_words.update(word.lower() for word in technical_words)
        
        # Add application-specific stop words
        if application:
            app_words = self.config.get('application_specific', {}).get(application.lower(), [])
            if isinstance(app_words, list):
                stop_words.update(word.lower() for word in app_words)
        
        # Cache the result
        self._stop_words_cache[cache_key] = stop_words
        
        logger.debug(f"Generated stop words set with {len(stop_words)} words for languages: {languages}, app: {application}")
        return stop_words
    
    def filter_words(self, 
                    words: List[str],
                    languages: Optional[List[str]] = None,
                    application: Optional[str] = None,
                    include_technical: bool = True) -> List[str]:
        """
        Filter a list of words by removing stop words.
        
        Args:
            words: List of words to filter
            languages: Languages to use for stop words
            application: Application-specific stop words to include
            include_technical: Whether to include technical stop words
            
        Returns:
            Filtered list of words
        """
        stop_words = self.get_stop_words(languages, application, include_technical)
        min_length = self.config.get('settings', {}).get('min_word_length', 3)
        case_sensitive = self.config.get('settings', {}).get('case_sensitive', False)
        
        filtered = []
        for word in words:
            # Apply case sensitivity
            check_word = word if case_sensitive else word.lower()
            
            # Filter by stop words and minimum length
            if (check_word not in stop_words and 
                len(word) >= min_length and
                word.isalpha()):  # Only keep alphabetic words
                filtered.append(word)
        
        return filtered
    
    def extract_key_terms(self, 
                         text: str,
                         languages: Optional[List[str]] = None,
                         application: Optional[str] = None,
                         max_terms: Optional[int] = None) -> List[str]:
        """
        Extract key terms from text using stop words filtering and frequency analysis.
        
        Args:
            text: Text to extract key terms from
            languages: Languages to use for stop words
            application: Application-specific stop words to include
            max_terms: Maximum number of terms to return
            
        Returns:
            List of key terms sorted by frequency
        """
        import re
        from collections import Counter
        
        if max_terms is None:
            max_terms = self.config.get('settings', {}).get('max_key_terms', 10)
        
        # Tokenize text (support for accented characters)
        words = re.findall(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]{3,}\b', text.lower())
        
        # Filter stop words
        filtered_words = self.filter_words(
            words, 
            languages=languages, 
            application=application,
            include_technical=True
        )
        
        # Count frequencies and return top terms
        counter = Counter(filtered_words)
        key_terms = [term for term, count in counter.most_common(max_terms)]
        
        logger.debug(f"Extracted {len(key_terms)} key terms from {len(words)} total words")
        return key_terms
    
    def get_settings(self) -> Dict[str, Any]:
        """Get processing settings from configuration."""
        return self.config.get('settings', {})
    
    def reload_config(self) -> None:
        """Reload configuration from file and clear cache."""
        self._stop_words_cache.clear()
        self._load_config()
        logger.info("Stop words configuration reloaded")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded stop words."""
        stats = {
            'config_loaded': bool(self.config),
            'nltk_available': NLTK_AVAILABLE,
            'nltk_initialized': self._nltk_initialized,
            'cache_size': len(self._stop_words_cache),
            'languages_configured': list(self.config.get('stop_words', {}).keys()),
            'applications_configured': list(self.config.get('application_specific', {}).keys())
        }
        
        # Count stop words by language
        for lang in stats['languages_configured']:
            lang_words = self.config.get('stop_words', {}).get(lang, [])
            stats[f'{lang}_stop_words_count'] = len(lang_words) if isinstance(lang_words, list) else 0
        
        return stats


# Global instance for easy access
_stop_words_manager = None

def get_stop_words_manager(config_path: str = "config/stop_words_config.yaml") -> StopWordsManager:
    """
    Get global stop words manager instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        StopWordsManager instance
    """
    global _stop_words_manager
    if _stop_words_manager is None:
        _stop_words_manager = StopWordsManager(config_path)
    return _stop_words_manager


# Convenience functions
def get_stop_words(languages: Optional[List[str]] = None,
                  application: Optional[str] = None,
                  include_technical: bool = True) -> Set[str]:
    """Convenience function to get stop words."""
    manager = get_stop_words_manager()
    return manager.get_stop_words(languages, application, include_technical)


def filter_words(words: List[str],
                languages: Optional[List[str]] = None,
                application: Optional[str] = None,
                include_technical: bool = True) -> List[str]:
    """Convenience function to filter words."""
    manager = get_stop_words_manager()
    return manager.filter_words(words, languages, application, include_technical)


def extract_key_terms(text: str,
                     languages: Optional[List[str]] = None,
                     application: Optional[str] = None,
                     max_terms: Optional[int] = None) -> List[str]:
    """Convenience function to extract key terms."""
    manager = get_stop_words_manager()
    return manager.extract_key_terms(text, languages, application, max_terms)


if __name__ == "__main__":
    # Test the stop words manager
    manager = StopWordsManager()
    
    print("Stop Words Manager Test")
    print("=" * 40)
    
    # Print stats
    stats = manager.get_stats()
    print("Configuration Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTesting key term extraction:")
    test_text = """
    This is a test document about SAP HANA database system.
    The system provides advanced analytics and real-time processing capabilities.
    Users can configure the database settings through various configuration files.
    """
    
    key_terms = manager.extract_key_terms(test_text, application='sap')
    print(f"Key terms: {key_terms}")
    
    print("\nTesting stop words for different configurations:")
    
    # English only
    en_stop_words = manager.get_stop_words(['english'])
    print(f"English stop words: {len(en_stop_words)} words")
    
    # Spanish only
    es_stop_words = manager.get_stop_words(['spanish'])
    print(f"Spanish stop words: {len(es_stop_words)} words")
    
    # Both languages + SAP application
    combined_stop_words = manager.get_stop_words(['english', 'spanish'], 'sap')
    print(f"Combined (EN+ES+SAP) stop words: {len(combined_stop_words)} words")
