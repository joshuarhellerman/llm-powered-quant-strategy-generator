"""
Strategy Extractor module for using LLMs to extract trading strategies from papers
Optimized for token budget and dry run support
"""

import os
import re
import json
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np


class StrategyExtractor:
    """
    Extracts trading strategies from research papers using LLMs
    """

    def __init__(self, llm_service, validation_service=None, config=None, output_dir="output/strategies_llm", test_mode=False):
        """
        Initialize the StrategyExtractor

        Args:
            llm_service: LLM service for extracting strategies
            validation_service: Service for validating strategy implementations
            config (dict): Configuration dictionary
            output_dir (str): Directory to save extracted strategies
            test_mode (bool): Whether to run in test mode with minimal API calls
        """
        self.llm_service = llm_service
        self.validation_service = validation_service
        self.config = config or {}
        self.output_dir = output_dir
        self.test_mode = test_mode

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Get token budget limits from config
        self.token_budget = self.config.get("llm", {}).get("token_budget", {})

        # Get dry run setting from LLM service
        self.dry_run = getattr(self.llm_service, "dry_run", False)
        if self.dry_run:
            self.logger.info("🔍 DRY RUN MODE ENABLED - Simulating LLM calls without API requests")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Subdirectories for different outputs
        self.overview_dir = os.path.join(output_dir, "overviews")
        self.code_dir = os.path.join(output_dir, "implementations")
        self.validation_dir = os.path.join(output_dir, "validation_reports")

        os.makedirs(self.overview_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)
        os.makedirs(self.validation_dir, exist_ok=True)

    def _validate_code(self, code, paper_id):
        """
        Validate generated code using the validation service

        Args:
            code (str): Python code to validate
            paper_id (str): ID of the paper

        Returns:
            tuple: (is_valid, issues)
        """
        if not self.validation_service:
            self.logger.warning("No validation service provided, skipping validation")
            return True, []

        try:
            self.logger.info(f"Validating code for paper {paper_id}")
            validation_result = self.validation_service.validate(code)

            is_valid = validation_result.get("is_valid", False)
            issues = validation_result.get("issues", [])

            # Save validation report
            report_file = os.path.join(self.validation_dir, f"{paper_id}_validation.json")
            with open(report_file, "w") as f:
                json.dump(validation_result, f, indent=2)

            return is_valid, issues

        except Exception as e:
            self.logger.error(f"Error validating code: {e}")
            return False, [f"Validation error: {e}"]

    def _validate_format_compliance(self, code):
        """
        Validate if the generated code complies with the standard format

        Args:
            code (str): Generated strategy code

        Returns:
            tuple: (is_compliant, issues)
        """
        issues = []

        # Check for BaseStrategy inheritance
        if "class" in code and "BaseStrategy" not in code:
            issues.append("Strategy class does not inherit from BaseStrategy")

        # Check for required methods
        required_methods = ["_initialize_parameters", "generate_signals"]
        for method in required_methods:
            if f"def {method}" not in code:
                issues.append(f"Missing required method: {method}")

        # Check for imports
        if "from strategy_format.base_strategy import BaseStrategy" not in code:
            issues.append("Missing import for BaseStrategy")

        # Check for indentation issues
        try:
            # Test if the code compiles
            compile(code, '<string>', 'exec')
        except IndentationError as e:
            issues.append(f"Indentation error in code at line {e.lineno}: {e.msg}")
        except SyntaxError as e:
            issues.append(f"Syntax error in code at line {e.lineno}: {e.msg}")

        return len(issues) == 0, issues

    def _get_class_name_from_code(self, code):
        """
        Extract class name from generated code

        Args:
            code (str): Python code

        Returns:
            str: Class name
        """
        # Find class definitions in the code
        class_matches = re.findall(r'class\s+(\w+)[\(:]', code)

        if class_matches:
            return class_matches[0]

        return "TradingStrategy"

    def _parse_formula_from_overview(self, strategy_overview):
        """
        Parse mathematical formulas from strategy overview

        Args:
            strategy_overview (dict): Strategy overview

        Returns:
            list: Parsed formulas
        """
        formulas = []

        if "key_formulas" in strategy_overview and isinstance(strategy_overview["key_formulas"], list):
            for formula in strategy_overview["key_formulas"]:
                if isinstance(formula, dict):
                    formulas.append({
                        "description": formula.get("description", ""),
                        "latex": formula.get("latex", ""),
                        "python": formula.get("python_equivalent", "")
                    })

        return formulas

    def _check_token_budget(self, operation_type="extract"):
        """
        Check if there's enough budget left for the operation

        Args:
            operation_type (str): Type of operation (extract, implement, refine)

        Returns:
            bool: True if operation is allowed, False if budget exceeded
        """
        # Skip check if dry run mode is enabled
        if self.dry_run:
            return True

        # Skip check if no token budget is set
        if not self.token_budget:
            return True

        # Get current token usage from LLM service
        token_usage = getattr(self.llm_service, "token_usage", {})
        total_cost = token_usage.get("estimated_cost", 0.0)

        # Get budget limits
        max_cost_per_call = self.token_budget.get("max_cost_per_call", float('inf'))
        total_budget = self.token_budget.get("total_budget", float('inf'))

        # Estimate cost for this operation
        estimated_operation_costs = {
            "extract": 0.05,      # Estimated cost for extraction
            "implement": 0.10,    # Estimated cost for implementation
            "refine": 0.05        # Estimated cost for refinement
        }

        operation_cost = estimated_operation_costs.get(operation_type, 0.05)

        # Check if operation would exceed budget
        if operation_cost > max_cost_per_call:
            self.logger.warning(f"⚠️ Operation '{operation_type}' estimated cost (${operation_cost:.4f}) exceeds max cost per call (${max_cost_per_call:.4f})")
            return False

        if total_cost + operation_cost > total_budget:
            self.logger.warning(f"⚠️ Budget limit reached: Current ${total_cost:.4f} + Operation ${operation_cost:.4f} would exceed total budget ${total_budget:.4f}")
            return False

        return True

    def extract_strategy(self, paper):
        """
        Extract a trading strategy from a paper

        Args:
            paper (dict): Paper dictionary with metadata

        Returns:
            dict: Extracted strategy information
        """
        paper_id = paper.get("id", "unknown")
        self.logger.info(f"Extracting strategy from paper {paper_id}")

        # Check if we have enough budget for extraction
        if not self._check_token_budget("extract"):
            self.logger.warning(f"Skipping strategy extraction for paper {paper_id} due to budget constraints")
            return {
                "paper_id": paper_id,
                "paper_title": paper.get("title", ""),
                "error": "Skipped due to token budget constraints",
                "is_valid": False
            }

        # Step 1: Extract strategy overview using LLM
        strategy_overview = self.llm_service.extract_strategy(paper)

        # Save strategy overview
        overview_file = os.path.join(self.overview_dir, f"{paper_id}_overview.json")
        with open(overview_file, "w") as f:
            json.dump(strategy_overview, f, indent=2)

        # In test mode or if budget is constrained, skip implementation
        if self.test_mode and not self.dry_run:
            self.logger.info(f"Test mode: Skipping implementation for paper {paper_id}")
            return {
                "paper_id": paper_id,
                "paper_title": paper.get("title", ""),
                "strategy_overview": strategy_overview,
                "implementation": "Skipped in test mode",
                "class_name": "TestStrategy",
                "is_valid": False,
                "issues": ["Implementation skipped in test mode"],
                "formulas": self._parse_formula_from_overview(strategy_overview),
                "code_file": None
            }

        # Check if we have enough budget for implementation
        if not self._check_token_budget("implement"):
            self.logger.warning(f"Skipping implementation for paper {paper_id} due to budget constraints")
            return {
                "paper_id": paper_id,
                "paper_title": paper.get("title", ""),
                "strategy_overview": strategy_overview,
                "implementation": "Skipped due to budget constraints",
                "class_name": "BudgetConstrainedStrategy",
                "is_valid": False,
                "issues": ["Implementation skipped due to budget constraints"],
                "formulas": self._parse_formula_from_overview(strategy_overview),
                "code_file": None
            }

        # Step 2: Generate implementation using LLM
        implementation = self.llm_service.generate_implementation(strategy_overview)

        # Fix indentation issues in dry run mode
        if self.dry_run:
            try:
                # Try to compile the code to check for syntax errors
                compile(implementation, '<string>', 'exec')
            except IndentationError:
                # If there's an indentation error, try to fix it
                self.logger.info("Fixing indentation issues in mock implementation")
                fixed_lines = []
                for i, line in enumerate(implementation.splitlines()):
                    # Strip leading whitespace and re-indent properly
                    stripped = line.lstrip()
                    if stripped.startswith('class '):
                        # Class definition should have no indent
                        fixed_lines.append(stripped)
                    elif stripped.startswith('def '):
                        # Method definitions should have 4 spaces
                        fixed_lines.append('    ' + stripped)
                    elif stripped and i > 0:
                        # Content inside methods should have 8 spaces
                        # But we need to check the context
                        prev_line = implementation.splitlines()[i - 1].lstrip()
                        if prev_line.startswith('def ') or prev_line.endswith(':'):
                            fixed_lines.append('        ' + stripped)
                        else:
                            fixed_lines.append('    ' + stripped)
                    else:
                        # Empty lines or imports
                        fixed_lines.append(stripped)
                implementation = '\n'.join(fixed_lines)

        # Check format compliance
        is_compliant, format_issues = self._validate_format_compliance(implementation)

        # If not compliant and we have budget, refinement is needed
        if not is_compliant and self._check_token_budget("refine"):
            self.logger.info(f"Strategy doesn't comply with format: {format_issues}")
            implementation = self.llm_service.refine_implementation(implementation, format_issues)

            # Check compliance again
            is_compliant, format_issues = self._validate_format_compliance(implementation)

        # Get class name from implementation
        class_name = self._get_class_name_from_code(implementation)

        # Save implementation
        code_file = os.path.join(self.code_dir, f"{paper_id}_{class_name}.py")
        with open(code_file, "w") as f:
            f.write(implementation)

        # Check if we have enough budget for implementation
        if not self._check_token_budget("implement"):
            self.logger.warning(f"Skipping implementation for paper {paper_id} due to budget constraints")
            return {
                "paper_id": paper_id,
                "paper_title": paper.get("title", ""),
                "strategy_overview": strategy_overview,
                "implementation": "Skipped due to budget constraints",
                "class_name": "BudgetConstrainedStrategy",
                "is_valid": False,
                "issues": ["Implementation skipped due to budget constraints"],
                "formulas": self._parse_formula_from_overview(strategy_overview),
                "code_file": None
            }

        # Step 2: Generate implementation using LLM
        implementation = self.llm_service.generate_implementation(strategy_overview)

        # Check format compliance
        is_compliant, format_issues = self._validate_format_compliance(implementation)

        # If not compliant and we have budget, refinement is needed
        if not is_compliant and self._check_token_budget("refine"):
            self.logger.info(f"Strategy doesn't comply with format: {format_issues}")
            implementation = self.llm_service.refine_implementation(implementation, format_issues)

            # Check compliance again
            is_compliant, format_issues = self._validate_format_compliance(implementation)

        # Get class name from implementation
        class_name = self._get_class_name_from_code(implementation)

        # Save implementation
        code_file = os.path.join(self.code_dir, f"{paper_id}_{class_name}.py")
        with open(code_file, "w") as f:
            f.write(implementation)

        # Step 3: Validate implementation if not in dry run mode
        if not self.dry_run:
            is_valid, issues = self._validate_code(implementation, paper_id)
        else:
            # Skip validation in dry run mode
            is_valid, issues = True, []

        # Step 4: Refine implementation if needed and budget allows
        if not is_valid and issues and self._check_token_budget("refine"):
            self.logger.info(f"Refining implementation for paper {paper_id}")
            implementation = self.llm_service.refine_implementation(implementation, issues)

            # Save refined implementation
            refined_code_file = os.path.join(self.code_dir, f"{paper_id}_{class_name}_refined.py")
            with open(refined_code_file, "w") as f:
                f.write(implementation)

            # Validate again if not in dry run mode
            if not self.dry_run:
                is_valid, issues = self._validate_code(implementation, paper_id)

        # Extract formulas from overview
        formulas = self._parse_formula_from_overview(strategy_overview)

        # Create result
        result = {
            "paper_id": paper_id,
            "paper_title": paper.get("title", ""),
            "strategy_overview": strategy_overview,
            "implementation": implementation,
            "class_name": class_name,
            "is_valid": is_valid,
            "issues": issues,
            "formulas": formulas,
            "code_file": code_file
        }

        return result

    def extract_strategies(self, papers):
        """
        Extract strategies from multiple papers

        Args:
            papers (list): List of paper dictionaries

        Returns:
            list: List of extracted strategy information
        """
        if not papers:
            self.logger.warning("No papers provided")
            return []

        # In test mode, limit to just 1 paper
        if self.test_mode and len(papers) > 1:
            self.logger.info(f"Test mode: Limiting to 1 paper instead of {len(papers)}")
            papers = papers[:1]
        else:
            self.logger.info(f"Extracting strategies from {len(papers)} papers")

        results = []
        for paper in tqdm(papers, desc="Extracting strategies"):
            try:
                result = self.extract_strategy(paper)
                results.append(result)

                # Check token usage after each paper
                if hasattr(self.llm_service, "get_token_usage_report"):
                    usage = self.llm_service.get_token_usage_report()
                    self.logger.info(f"Current token usage: {usage['total_tokens']} tokens (${usage['estimated_cost']:.4f})")

                    # Check if we've exceeded our total budget
                    total_budget = self.token_budget.get("total_budget", float('inf'))
                    if usage['estimated_cost'] >= total_budget:
                        self.logger.warning(f"⚠️ Total budget of ${total_budget:.2f} exceeded. Stopping extraction.")
                        break

            except Exception as e:
                self.logger.error(f"Error extracting strategy from paper {paper.get('id', 'unknown')}: {e}")

        # Save overall results
        results_file = os.path.join(self.output_dir, "extraction_results.json")

        # Create a simplified version for JSON serialization
        simplified_results = []
        for result in results:
            simplified_result = {
                "paper_id": result["paper_id"],
                "paper_title": result["paper_title"],
                "class_name": result["class_name"],
                "is_valid": result["is_valid"],
                "code_file": result["code_file"],
                "strategy_name": result.get("strategy_overview", {}).get("strategy_name", ""),
                "core_mechanism": result.get("strategy_overview", {}).get("core_mechanism", "")[:100] + "...",
                "asset_classes": result.get("strategy_overview", {}).get("asset_classes", []),
                "num_formulas": len(result.get("formulas", []))
            }
            simplified_results.append(simplified_result)

        with open(results_file, "w") as f:
            json.dump(simplified_results, f, indent=2)

        # Generate token usage report if available
        if hasattr(self.llm_service, "get_token_usage_report"):
            usage_report = self.llm_service.get_token_usage_report()
            usage_file = os.path.join(self.output_dir, "token_usage_report.json")
            with open(usage_file, "w") as f:
                json.dump(usage_report, f, indent=2)

            self.logger.info(f"Token usage report: {usage_report['total_tokens']} tokens, ${usage_report['estimated_cost']:.4f}")

        self.logger.info(f"Extracted {len(results)} strategies")
        return results

    def test_strategy_viability(self, strategy_result, test_data=None):
        """
        Test if a strategy implementation is viable with real data

        Args:
            strategy_result (dict): Extracted strategy result
            test_data (pandas.DataFrame): Optional test data

        Returns:
            dict: Test results
        """
        if not strategy_result["is_valid"]:
            self.logger.warning(f"Strategy {strategy_result['class_name']} is not valid, skipping test")
            return {"is_viable": False, "reason": "Strategy is not valid"}

        # Get implementation code
        code = strategy_result["implementation"]
        class_name = strategy_result["class_name"]

        try:
            # Create a temporary file with the implementation
            temp_file = os.path.join(self.output_dir, f"temp_{class_name}.py")
            with open(temp_file, "w") as f:
                f.write(code)

            # Import the strategy class
            import importlib.util
            spec = importlib.util.spec_from_file_location(class_name, temp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the strategy class
            strategy_class = getattr(module, class_name)

            # Create an instance of the strategy
            strategy = strategy_class()

            # Generate test data if not provided
            if test_data is None:
                test_data = self._generate_test_data()

            # Test the strategy with the data
            try:
                # Try to run backtest method if available
                if hasattr(strategy, "backtest") and callable(getattr(strategy, "backtest")):
                    backtest_data, metrics = strategy.backtest(test_data)

                    return {
                        "is_viable": True,
                        "metrics": metrics,
                        "strategy": strategy,
                        "backtest_data": backtest_data
                    }

                # If no backtest method, try to run generate_signals
                elif hasattr(strategy, "generate_signals") and callable(getattr(strategy, "generate_signals")):
                    signals_data = strategy.generate_signals(test_data)

                    return {
                        "is_viable": True,
                        "strategy": strategy,
                        "signals_data": signals_data
                    }

                else:
                    return {"is_viable": False, "reason": "No backtest or generate_signals method found"}

            except Exception as e:
                self.logger.error(f"Error testing strategy {class_name}: {e}")
                return {"is_viable": False, "reason": f"Error testing strategy: {e}"}

        except Exception as e:
            self.logger.error(f"Error importing strategy {class_name}: {e}")
            return {"is_viable": False, "reason": f"Error importing strategy: {e}"}

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _generate_test_data(self, days=500):
        """
        Generate synthetic market data for testing strategies

        Args:
            days (int): Number of days of data to generate

        Returns:
            pandas.DataFrame: Synthetic market data
        """
        # Generate dates
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.Timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')

        # Generate price data using a random walk with drift
        np.random.seed(42)  # For reproducibility

        # Parameters
        initial_price = 100.0
        drift = 0.0001  # Small upward drift
        volatility = 0.015  # Daily volatility

        # Generate returns
        returns = np.random.normal(drift, volatility, size=len(dates))

        # Calculate price series
        log_returns = np.cumsum(returns)
        prices = initial_price * np.exp(log_returns)

        # Create dataframe
        df = pd.DataFrame({
            'open': prices * (1 - np.random.uniform(0, 0.005, size=len(dates))),
            'high': prices * (1 + np.random.uniform(0.001, 0.01, size=len(dates))),
            'low': prices * (1 - np.random.uniform(0.001, 0.01, size=len(dates))),
            'close': prices,
            'volume': np.random.randint(100000, 10000000, size=len(dates))
        }, index=dates)

        # Ensure high > open/close > low
        df['high'] = df[['high', 'open', 'close']].max(axis=1) * (1 + 0.001)
        df['low'] = df[['low', 'open', 'close']].min(axis=1) * (1 - 0.001)

        return df

    def create_strategy_catalog(self, strategies):
        """
        Create a catalog of strategies with their properties

        Args:
            strategies (list): List of extracted strategy results

        Returns:
            pandas.DataFrame: Catalog of strategies
        """
        catalog_data = []

        for strat in strategies:
            overview = strat.get("strategy_overview", {})

            entry = {
                "paper_id": strat.get("paper_id", ""),
                "paper_title": strat.get("paper_title", ""),
                "strategy_name": overview.get("strategy_name", ""),
                "class_name": strat.get("class_name", ""),
                "is_valid": strat.get("is_valid", False),
                "core_mechanism": overview.get("core_mechanism", "")[:200],
                "asset_classes": ", ".join(overview.get("asset_classes", [])),
                "market_conditions": ", ".join(overview.get("market_conditions", [])),
                "time_frames": ", ".join(overview.get("time_frames", [])),
                "num_indicators": len(overview.get("indicators", [])),
                "num_formulas": len(strat.get("formulas", [])),
                "code_file": strat.get("code_file", "")
            }

            catalog_data.append(entry)

        # Create DataFrame
        catalog = pd.DataFrame(catalog_data)

        # Save catalog
        catalog_file = os.path.join(self.output_dir, "strategy_catalog.csv")
        catalog.to_csv(catalog_file, index=False)

        return catalog