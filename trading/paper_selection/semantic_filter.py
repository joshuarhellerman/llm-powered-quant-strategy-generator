import re
import json
import os
import logging


class SemanticFilter:
    """
    Uses Claude to semantically evaluate papers for
    complete, implementable trading strategies.
    """

    def __init__(self, llm_service, dry_run=False):
        self.llm_service = llm_service
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)

        # Cache for semantic evaluations to avoid repeat API calls
        self.cache_dir = "cache/semantic_filter"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache = {}
        self._load_cache()

        if self.dry_run:
            self.logger.info("SemanticFilter initialized in dry run mode - will not make LLM calls")

    def evaluate_paper(self, paper):
        """
        Evaluate a paper semantically for trading strategy completeness

        Args:
            paper (dict): Paper dictionary

        Returns:
            tuple: (score, analysis) where score is a float and analysis is a dict
        """
        paper_id = paper.get('id', 'unknown')

        # Check cache first
        if paper_id in self.cache:
            self.logger.info(f"Using cached semantic evaluation for paper {paper_id}")
            return self.cache[paper_id]['score'], self.cache[paper_id]['analysis']

        # In dry run mode, return mock evaluation
        if self.dry_run:
            self.logger.info(f"Dry run mode: Returning mock evaluation for paper {paper_id}")
            return self._get_mock_evaluation(paper)

        # Extract content for analysis
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')

        # Get additional content if available
        content = paper.get('content', '')
        if not content and 'structure_analysis' in paper:
            # Extract relevant sections from structure analysis
            sections = paper['structure_analysis'].get('identified_sections', {})
            section_texts = []
            for section_title, category in sections.items():
                if category in ['trading_rules', 'backtest_results', 'risk_management']:
                    # This would be more complex in a real implementation
                    # requiring extraction of section text
                    section_texts.append(f"Section {section_title}")

            content = "\n\n".join(section_texts) if section_texts else abstract

        # Generate prompt for LLM
        prompt = self._create_evaluation_prompt(title, abstract, content)

        try:
            # Query LLM
            self.logger.info(f"Querying LLM for semantic evaluation of paper {paper_id}")
            response = self.llm_service.query(
                prompt,
                system_prompt=self._get_system_prompt(),
                max_tokens=1000
            )

            # Parse response
            score, analysis = self._parse_evaluation_response(response)

            # Cache the result
            self.cache[paper_id] = {
                'score': score,
                'analysis': analysis
            }
            self._save_cache()

            return score, analysis

        except Exception as e:
            self.logger.error(f"Error during LLM evaluation: {e}")
            # Return fallback evaluation
            return self._get_fallback_evaluation(paper)

    def _get_mock_evaluation(self, paper):
        """Generate mock evaluation for dry run mode"""
        # Create a mock evaluation based on paper characteristics
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()

        # Simple keyword-based scoring for mock evaluation
        strategy_keywords = ['trading', 'strategy', 'algorithmic', 'momentum', 'mean reversion']
        keyword_score = sum(1 for keyword in strategy_keywords if keyword in f"{title} {abstract}")

        # Normalize to 0-1 range
        mock_score = min(1.0, keyword_score / len(strategy_keywords) + 0.3)

        # Special handling for synthetic test papers
        if paper.get('_synthetic_test_paper', False):
            mock_score = 0.9

        mock_analysis = {
            'strategy_score': mock_score,
            'has_entry_rules': mock_score > 0.5,
            'has_exit_rules': mock_score > 0.5,
            'has_parameters': mock_score > 0.4,
            'has_position_sizing': mock_score > 0.6,
            'has_backtest': mock_score > 0.3,
            'is_prediction_only': mock_score < 0.4,
            'summary': f'Mock evaluation for dry run mode (score: {mock_score:.2f})',
            'key_elements': ['Mock trading strategy elements'] if mock_score > 0.5 else [],
            'missing_elements': ['Complete implementation details'] if mock_score < 0.8 else []
        }

        return mock_score, mock_analysis

    def _get_fallback_evaluation(self, paper):
        """Generate fallback evaluation when LLM fails"""
        # Use basic keyword analysis as fallback
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        text = f"{title} {abstract}"

        # Simple scoring based on keywords
        trading_words = ['trading', 'strategy', 'buy', 'sell', 'position']
        backtest_words = ['backtest', 'performance', 'sharpe', 'return']

        trading_score = sum(1 for word in trading_words if word in text) / len(trading_words)
        backtest_score = sum(1 for word in backtest_words if word in text) / len(backtest_words)

        fallback_score = (trading_score + backtest_score) / 2

        fallback_analysis = {
            'strategy_score': fallback_score,
            'has_entry_rules': 'buy' in text or 'long' in text,
            'has_exit_rules': 'sell' in text or 'short' in text,
            'has_parameters': 'threshold' in text or 'parameter' in text,
            'has_position_sizing': 'position' in text,
            'has_backtest': any(word in text for word in backtest_words),
            'is_prediction_only': 'predict' in text and 'trading' not in text,
            'summary': 'Fallback evaluation due to LLM error',
            'key_elements': ['Basic trading concepts found'],
            'missing_elements': ['Detailed analysis unavailable'],
            'fallback': True
        }

        return fallback_score, fallback_analysis

    def _create_evaluation_prompt(self, title, abstract, content):
        """Create prompt for LLM evaluation"""
        return f"""
        Evaluate whether this academic paper contains a complete, implementable trading strategy.

        Paper Title: {title}

        Abstract: {abstract}

        Additional Content: {content}

        A complete trading strategy requires:
        1. Specific entry and exit rules (not just a prediction model)
        2. Clear signal generation logic
        3. Explicit parameter values and thresholds
        4. Position sizing methodology
        5. Evidence of backtest evaluation

        Score the paper from 0.0 to 1.0 on its completeness as an implementable trading strategy.

        Output your evaluation in JSON format:

        ```json
        {{
          "strategy_score": 0.0-1.0,
          "has_entry_rules": true/false,
          "has_exit_rules": true/false,
          "has_parameters": true/false,
          "has_position_sizing": true/false,
          "has_backtest": true/false,
          "is_prediction_only": true/false,
          "summary": "Brief evaluation summary",
          "key_elements": ["List of key strategy elements found"],
          "missing_elements": ["List of elements needed to implement this as a strategy"]
        }}
        ```

        Only output valid JSON. Focus on the presence of actual trading rules, not just prediction models.
        """

    def _get_system_prompt(self):
        """Get system prompt for LLM"""
        return """
        You are an expert quant researcher specializing in evaluating academic papers 
        for implementable trading strategies. Your task is to distinguish between papers 
        that describe complete trading strategies versus those that only present prediction 
        models without explicit trading rules.

        A complete trading strategy must have:
        - Explicit buy/sell signals or entry/exit rules
        - Clear thresholds or parameters for signal generation
        - Position sizing approach (even if basic)

        Papers that only predict returns, volatility, or other market variables without 
        translating these predictions into trading decisions are NOT complete strategies.
        """

    def _parse_evaluation_response(self, response):
        """Parse the LLM response to extract score and analysis"""
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without the markdown code block
                json_match = re.search(r'({[\s\S]*})', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    raise ValueError("No JSON found in response")

            # Parse JSON
            analysis = json.loads(json_str)

            # Extract score
            score = analysis.get('strategy_score', 0.0)

            return score, analysis

        except Exception as e:
            self.logger.error(f"Failed to parse evaluation response: {e}")
            return 0.0, {"error": str(e), "raw_response": response}

    def _load_cache(self):
        """Load cache from disk"""
        cache_file = os.path.join(self.cache_dir, "semantic_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                self.logger.info(f"Loaded semantic evaluation cache with {len(self.cache)} entries")
            except Exception as e:
                self.logger.error(f"Failed to load semantic cache: {e}")
                self.cache = {}

    def _save_cache(self):
        """Save cache to disk"""
        cache_file = os.path.join(self.cache_dir, "semantic_cache.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save semantic cache: {e}")

    def clear_cache(self):
        """Clear the evaluation cache"""
        self.cache = {}
        cache_file = os.path.join(self.cache_dir, "semantic_cache.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        self.logger.info("Semantic evaluation cache cleared")

    def get_cache_stats(self):
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache),
            'cache_file': os.path.join(self.cache_dir, "semantic_cache.json")
        }