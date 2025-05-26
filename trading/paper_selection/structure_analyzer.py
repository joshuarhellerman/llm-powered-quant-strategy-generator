import re
import logging
import os
import json


class StructureAnalyzer:
    """
    Analyzes document structure to identify key sections
    relevant to trading strategies.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Section patterns to identify
        self.section_patterns = {
            'trading_rules': [
                r'(?i)trading rules', r'(?i)trading strategy', r'(?i)strategy rules',
                r'(?i)trading signals', r'(?i)entry and exit', r'(?i)position rules'
            ],
            'backtest_results': [
                r'(?i)backtest', r'(?i)empirical results', r'(?i)performance',
                r'(?i)out-of-sample', r'(?i)sharpe ratio', r'(?i)trading results'
            ],
            'risk_management': [
                r'(?i)risk management', r'(?i)position sizing', r'(?i)stop loss',
                r'(?i)risk control', r'(?i)drawdown', r'(?i)portfolio constraints'
            ],
            'mathematical_model': [
                r'(?i)mathematical model', r'(?i)formula', r'(?i)equation',
                r'(?i)algorithm', r'(?i)pseudocode', r'(?i)parameters'
            ]
        }

        # Rule patterns to identify
        self.rule_patterns = {
            'entry_rules': [
                r'(?i)buy when', r'(?i)enter (a|the) (long|short|trade)',
                r'(?i)if .{1,50} then buy', r'(?i)entry signal',
                r'(?i)entry condition', r'(?i)buy signal'
            ],
            'exit_rules': [
                r'(?i)sell when', r'(?i)exit (a|the) (long|short|trade)',
                r'(?i)if .{1,50} then sell', r'(?i)exit signal',
                r'(?i)exit condition', r'(?i)sell signal'
            ],
            'thresholds': [
                r'threshold of (\d+\.?\d*)', r'(\d+\.?\d*)% threshold',
                r'(greater|less) than (\d+\.?\d*)', r'exceeds (\d+\.?\d*)'
            ]
        }

    def analyze_structure(self, paper):
        """
        Analyze document structure to identify key sections

        Args:
            paper (dict): Paper dictionary

        Returns:
            dict: Structure analysis results
        """
        # Get paper content
        content = ""

        if 'content' in paper and paper['content']:
            content = paper['content']
        else:
            # If full content not available, use abstract
            content = paper.get('abstract', '')

        # Initialize results
        results = {
            'has_trading_rules': False,
            'has_backtest_results': False,
            'has_risk_management': False,
            'has_mathematical_model': False,
            'has_entry_rules': False,
            'has_exit_rules': False,
            'has_thresholds': False,
            'section_matches': {},
            'rule_matches': {},
            'extracted_thresholds': []
        }

        # Check for each section type
        for section_type, patterns in self.section_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(re.findall(pattern, content))

            results[f'has_{section_type}'] = len(matches) > 0
            results['section_matches'][section_type] = matches

        # Check for rule patterns
        for rule_type, patterns in self.rule_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(re.findall(pattern, content))

            results[f'has_{rule_type}'] = len(matches) > 0
            results['rule_matches'][rule_type] = matches

            # Extract thresholds from matches
            if rule_type == 'thresholds':
                results['extracted_thresholds'] = self._extract_threshold_values(matches)

        # Extract potential sections using header detection
        results['identified_sections'] = self._identify_sections(content)

        return results

    def _extract_threshold_values(self, matches):
        """Extract numeric threshold values from regex matches"""
        thresholds = []

        for match in matches:
            # Try to extract numbers from the match
            if isinstance(match, tuple):
                for item in match:
                    try:
                        value = float(item)
                        thresholds.append(value)
                    except (ValueError, TypeError):
                        pass
            else:
                # Try to find numbers in the string
                numbers = re.findall(r'(\d+\.?\d*)', match)
                for num in numbers:
                    try:
                        thresholds.append(float(num))
                    except (ValueError, TypeError):
                        pass

        return thresholds

    def _identify_sections(self, content):
        """Identify potential section headers in the content"""
        # This is a simplified approach - for PDFs we'd use more sophisticated methods
        sections = {}

        # Look for potential section headers (numbered or not)
        section_pattern = r'(?:^|\n)(?:\d+\.\d*\s+)?([A-Z][A-Za-z\s]{3,50})(?:\n|\.)'

        matches = re.findall(section_pattern, content)
        for section_title in matches:
            # Check if this section title contains keywords related to trading strategies
            title_lower = section_title.lower()

            # Categorize the section
            category = None
            for section_type, patterns in self.section_patterns.items():
                if any(re.search(pattern, title_lower) for pattern in patterns):
                    category = section_type
                    break

            if category:
                sections[section_title] = category

        return sections