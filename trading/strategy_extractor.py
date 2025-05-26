"""
Strategy Extractor with configurable cost controls
Extracts trading strategies from research papers using LLM
"""

import os
import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple


class StrategyExtractor:
    """
    Strategy Extractor with configurable cost controls and BaseStrategy compliance
    """

    def __init__(self, llm_service, validation_service=None, config=None, output_dir="output/strategies_llm", test_mode=False):
        """Initialize with cost control configuration"""
        self.llm_service = llm_service
        self.validation_service = validation_service
        self.config = config or {}
        self.output_dir = output_dir
        self.test_mode = test_mode
        self.logger = logging.getLogger(__name__)

        # Extract cost control settings
        self.llm_config = self.config.get("llm", {})
        self.implementation_config = self.llm_config.get("implementation", {})
        self.extraction_config = self.config.get("strategy_extraction", {})

        # Cost control flags
        self.implementation_enabled = self.implementation_config.get("enabled", True)
        self.skip_implementation_in_test = self.implementation_config.get("test_mode_skip", True)
        self.max_cost_per_implementation = self.implementation_config.get("max_cost_per_implementation", 0.50)

        # Budget controls
        self.token_budget = self.llm_config.get("token_budget", {})
        self.max_cost_per_call = self.token_budget.get("max_cost_per_call", 0.10)
        self.total_budget = self.token_budget.get("total_budget", 5.00)

        # Setup directories
        os.makedirs(output_dir, exist_ok=True)
        self.overview_dir = os.path.join(output_dir, "overviews")
        self.code_dir = os.path.join(output_dir, "implementations")
        self.validation_dir = os.path.join(output_dir, "validation_reports")

        for dir_path in [self.overview_dir, self.code_dir, self.validation_dir]:
            os.makedirs(dir_path, exist_ok=True)

        self.logger.info(f"StrategyExtractor initialized with output_dir: {output_dir}")

    def should_skip_implementation(self, paper_id: str) -> Tuple[bool, str]:
        """
        Determine if implementation should be skipped based on configuration

        Returns:
            tuple: (should_skip, reason)
        """
        # Check if implementation is globally disabled
        if not self.implementation_enabled:
            return True, "Implementation disabled in configuration"

        # Check test mode settings
        if self.test_mode and self.skip_implementation_in_test:
            return True, "Test mode: Implementation skipped to save costs"

        # Check if we've exceeded budget
        if hasattr(self.llm_service, 'token_usage'):
            current_cost = self.llm_service.token_usage.get('estimated_cost', 0.0)
            if current_cost >= self.total_budget:
                return True, f"Budget limit reached: ${current_cost:.4f} >= ${self.total_budget:.4f}"

        # Check if single implementation would exceed per-call limit
        if self.max_cost_per_implementation > self.max_cost_per_call:
            self.logger.warning(f"Implementation cost limit (${self.max_cost_per_implementation}) exceeds per-call limit (${self.max_cost_per_call})")

        return False, "Implementation approved"

    def estimate_implementation_cost(self, strategy_overview: Dict[str, Any]) -> float:
        """
        Estimate the cost of generating implementation for a strategy

        Returns:
            float: Estimated cost in USD
        """
        # Base cost estimate for implementation
        base_cost = 0.02  # Typical implementation cost

        # Adjust based on strategy complexity
        complexity_indicators = [
            len(strategy_overview.get('indicators', [])),
            len(strategy_overview.get('key_formulas', [])),
            len(str(strategy_overview.get('core_mechanism', ''))),
        ]

        complexity_factor = 1.0
        if any(indicator > 5 for indicator in complexity_indicators):
            complexity_factor = 1.5  # More complex strategy
        elif any(indicator > 10 for indicator in complexity_indicators):
            complexity_factor = 2.0  # Very complex strategy

        estimated_cost = base_cost * complexity_factor

        # Cap at configured maximum
        estimated_cost = min(estimated_cost, self.max_cost_per_implementation)

        return estimated_cost

    def extract_strategies(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract strategies from multiple papers

        Args:
            papers: List of paper dictionaries

        Returns:
            List of extraction results
        """
        results = []

        self.logger.info(f"Starting strategy extraction for {len(papers)} papers")

        for i, paper in enumerate(papers):
            paper_id = paper.get("id", f"paper_{i}")
            self.logger.info(f"Processing paper {i+1}/{len(papers)}: {paper_id}")

            try:
                result = self.extract_strategy(paper)
                results.append(result)

                # Log progress
                if result.get("is_valid", False):
                    self.logger.info(f"✅ Successfully extracted strategy from {paper_id}")
                else:
                    self.logger.warning(f"⚠️ Strategy extraction had issues for {paper_id}: {result.get('skip_reason', 'Unknown issue')}")

            except Exception as e:
                self.logger.error(f"❌ Failed to process paper {paper_id}: {e}")
                results.append({
                    "paper_id": paper_id,
                    "error": str(e),
                    "is_valid": False
                })

        self.logger.info(f"Strategy extraction complete. {len([r for r in results if r.get('is_valid', False)])} valid strategies extracted")
        return results

    def extract_strategy(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract strategy with configurable cost controls
        """
        paper_id = paper.get("id", "unknown")
        paper_title = paper.get("title", "")

        self.logger.info(f"Extracting strategy from paper {paper_id}")

        # Step 1: Always extract strategy overview (low cost)
        self.logger.info(f"Step 1: Extracting strategy overview for {paper_id}")

        try:
            strategy_overview = self.llm_service.extract_strategy(paper)

            # Save strategy overview
            overview_file = os.path.join(self.overview_dir, f"{paper_id}_overview.json")
            with open(overview_file, "w") as f:
                json.dump(strategy_overview, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to extract strategy overview for {paper_id}: {e}")
            return {
                "paper_id": paper_id,
                "paper_title": paper_title,
                "error": f"Overview extraction failed: {e}",
                "is_valid": False
            }

        # Step 2: Decide whether to generate implementation
        should_skip, skip_reason = self.should_skip_implementation(paper_id)

        if should_skip:
            self.logger.info(f"Step 2: Skipping implementation for {paper_id} - {skip_reason}")
            return {
                "paper_id": paper_id,
                "paper_title": paper_title,
                "strategy_overview": strategy_overview,
                "implementation": f"Implementation skipped: {skip_reason}",
                "class_name": "SkippedStrategy",
                "is_valid": False,
                "skip_reason": skip_reason,
                "code_file": None
            }

        # Step 3: Check cost estimate
        estimated_cost = self.estimate_implementation_cost(strategy_overview)
        self.logger.info(f"Step 3: Estimated implementation cost for {paper_id}: ${estimated_cost:.4f}")

        if estimated_cost > self.max_cost_per_implementation:
            skip_reason = f"Estimated cost ${estimated_cost:.4f} exceeds limit ${self.max_cost_per_implementation:.4f}"
            self.logger.warning(f"Skipping implementation for {paper_id}: {skip_reason}")
            return {
                "paper_id": paper_id,
                "paper_title": paper_title,
                "strategy_overview": strategy_overview,
                "implementation": f"Implementation skipped: {skip_reason}",
                "class_name": "CostLimitedStrategy",
                "is_valid": False,
                "skip_reason": skip_reason,
                "code_file": None
            }

        # Step 4: Generate implementation
        self.logger.info(f"Step 4: Generating implementation for {paper_id} (estimated cost: ${estimated_cost:.4f})")

        try:
            implementation = self.llm_service.generate_implementation(strategy_overview)

            # Get class name from implementation
            class_name = self._get_class_name_from_code(implementation)

            # Save implementation
            code_file = os.path.join(self.code_dir, f"{paper_id}_{class_name}.py")
            with open(code_file, "w") as f:
                f.write(implementation)

            self.logger.info(f"Successfully saved implementation to: {code_file}")

            # Step 5: Validate if enabled
            if self.extraction_config.get("require_validation", True):
                self.logger.info(f"Step 5: Validating implementation for {paper_id}")
                is_valid, issues = self._validate_implementation(implementation, paper_id)
            else:
                is_valid, issues = True, []

            return {
                "paper_id": paper_id,
                "paper_title": paper_title,
                "strategy_overview": strategy_overview,
                "implementation": implementation,
                "class_name": class_name,
                "is_valid": is_valid,
                "issues": issues,
                "code_file": code_file,
                "estimated_cost": estimated_cost
            }

        except Exception as e:
            self.logger.error(f"Failed to generate implementation for {paper_id}: {e}")
            return {
                "paper_id": paper_id,
                "paper_title": paper_title,
                "strategy_overview": strategy_overview,
                "implementation": f"Implementation failed: {e}",
                "class_name": "FailedStrategy",
                "is_valid": False,
                "error": str(e),
                "code_file": None
            }

    def _get_class_name_from_code(self, code: str) -> str:
        """Extract class name from generated code"""
        class_matches = re.findall(r'class\s+(\w+)[\(:]', code)
        return class_matches[0] if class_matches else "GeneratedStrategy"

    def _validate_format_compliance(self, code: str) -> Tuple[bool, List[str]]:
        """Enhanced validation for BaseStrategy compliance"""
        issues = []

        # Check for proper imports
        if "from base_strategy import BaseStrategy" not in code:
            issues.append("Missing import: 'from base_strategy import BaseStrategy'")

        # Check for class inheritance
        if not re.search(r'class\s+\w+\(BaseStrategy\):', code):
            issues.append("Class must inherit from BaseStrategy: 'class YourStrategy(BaseStrategy):'")

        # Check for required abstract methods
        if not re.search(r'def _initialize_parameters\(self, params\):', code):
            issues.append("Missing required method: '_initialize_parameters(self, params)'")

        if not re.search(r'def generate_signals\(self, data\):', code):
            issues.append("Missing required method: 'generate_signals(self, data)'")

        # Check that forbidden methods are not implemented
        forbidden_methods = ['backtest', 'plot_results', '_calculate_metrics']
        for method in forbidden_methods:
            if re.search(rf'def {method}\(', code):
                issues.append(f"Do not implement '{method}' method - it's provided by BaseStrategy")

        # Check for signal column initialization
        if "df['signal'] = 0" not in code and "signal" in code:
            issues.append("Must initialize signal column with: df['signal'] = 0")

        # Test syntax
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        except IndentationError as e:
            issues.append(f"Indentation error: {e}")

        return len(issues) == 0, issues

    def _validate_implementation(self, code: str, paper_id: str) -> Tuple[bool, List[str]]:
        """Validate generated implementation"""
        if not self.validation_service:
            # Use built-in format validation if no validation service
            return self._validate_format_compliance(code)

        try:
            validation_result = self.validation_service.validate(code)
            is_valid = validation_result.get("is_valid", False)
            issues = validation_result.get("issues", [])

            # Save validation report
            report_file = os.path.join(self.validation_dir, f"{paper_id}_validation.json")
            with open(report_file, "w") as f:
                json.dump(validation_result, f, indent=2)

            return is_valid, issues

        except Exception as e:
            self.logger.error(f"Validation failed for {paper_id}: {e}")
            # Fallback to format validation
            return self._validate_format_compliance(code)

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get a summary of costs and budget usage"""
        if not hasattr(self.llm_service, 'token_usage'):
            return {"error": "No token usage data available"}

        token_usage = self.llm_service.token_usage

        return {
            "total_cost": token_usage.get('estimated_cost', 0.0),
            "budget_limit": self.total_budget,
            "budget_remaining": self.total_budget - token_usage.get('estimated_cost', 0.0),
            "budget_used_percent": (token_usage.get('estimated_cost', 0.0) / self.total_budget) * 100,
            "implementation_enabled": self.implementation_enabled,
            "test_mode_skip": self.skip_implementation_in_test,
            "max_cost_per_implementation": self.max_cost_per_implementation
        }

    def create_fallback_strategy(self) -> str:
        """Create a robust fallback strategy with proper BaseStrategy inheritance"""
        self.logger.info("Creating robust fallback strategy...")

        fallback_code = '''import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'strategy_format'))
from base_strategy import BaseStrategy
import pandas as pd
import numpy as np


class FallbackMovingAverageStrategy(BaseStrategy):
    """
    Robust fallback strategy when LLM extraction fails
    Simple moving average crossover strategy with proper BaseStrategy inheritance
    """

    def _initialize_parameters(self, params):
        """Initialize strategy parameters"""
        self.fast_period = params.get('fast_period', 10)
        self.slow_period = params.get('slow_period', 50)
        self.name = "Fallback Moving Average Strategy"
        self.description = "Simple moving average crossover strategy"

    def generate_signals(self, data):
        """Generate trading signals based on moving average crossover"""
        df = data.copy()

        # Ensure we have enough data
        if len(df) < max(self.fast_period, self.slow_period):
            df['signal'] = 0
            return df

        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_period, min_periods=1).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period, min_periods=1).mean()

        # Generate signals
        df['signal'] = 0

        # Buy when fast MA > slow MA
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1

        # Sell when fast MA < slow MA
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1

        # Ensure no NaN values in signal column
        df['signal'] = df['signal'].fillna(0)

        return df
'''

        fallback_file = os.path.join(self.code_dir, "fallback_strategy.py")
        with open(fallback_file, 'w') as f:
            f.write(fallback_code)

        self.logger.info(f"✅ Created robust fallback strategy: {fallback_file}")
        return fallback_file


# Example configurations for different use cases
class StrategyExtractorConfig:
    """Configuration helper for StrategyExtractor"""

    @staticmethod
    def development_config():
        """Development: Extract overviews only, skip implementation"""
        return {
            "llm": {
                "implementation": {
                    "enabled": True,
                    "test_mode_skip": True,  # Skip in test mode
                    "max_cost_per_implementation": 0.10
                },
                "token_budget": {
                    "total_budget": 1.00  # Low budget for development
                }
            },
            "strategy_extraction": {
                "extract_implementation": False,  # Skip implementation
                "use_cache": True,
                "require_validation": False
            }
        }

    @staticmethod
    def production_config():
        """Production: Full extraction with higher budget"""
        return {
            "llm": {
                "implementation": {
                    "enabled": True,
                    "test_mode_skip": False,  # Generate even in test mode
                    "max_cost_per_implementation": 0.50
                },
                "token_budget": {
                    "total_budget": 20.00  # Higher budget
                }
            },
            "strategy_extraction": {
                "extract_implementation": True,
                "max_refinement_attempts": 3,
                "require_validation": True
            }
        }

    @staticmethod
    def budget_conscious_config():
        """Budget-conscious: Overview only"""
        return {
            "llm": {
                "implementation": {
                    "enabled": False,  # No implementation at all
                },
                "token_budget": {
                    "total_budget": 0.50  # Very low budget
                }
            },
            "strategy_extraction": {
                "extract_implementation": False,
                "require_validation": False
            }
        }