"""
Validation service for checking strategy implementations
"""

import os
import re
import ast
import logging
import tempfile
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class ValidationService:
    """
    Service for validating trading strategy implementations
    """

    def __init__(self, config=None):
        """
        Initialize the ValidationService

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Load test data if available
        self.test_data = None
        test_data_path = self.config.get("test_data_path")
        if test_data_path and os.path.exists(test_data_path):
            try:
                self.test_data = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
                self.logger.info(f"Loaded test data from {test_data_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load test data: {e}")

    def validate_syntax(self, code):
        """
        Validate Python syntax

        Args:
            code (str): Python code to validate

        Returns:
            tuple: (is_valid, issues)
        """
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            return False, [f"Syntax error at line {e.lineno}: {e.msg}"]

    def validate_imports(self, code):
        """
        Validate imports in the code

        Args:
            code (str): Python code to validate

        Returns:
            tuple: (is_valid, issues)
        """
        issues = []

        # Check for common data science libraries
        required_imports = ["pandas", "numpy", "matplotlib"]

        for imp in required_imports:
            if not re.search(fr'\bimport\s+{imp}\b', code) and not re.search(fr'\bfrom\s+{imp}\b', code):
                issues.append(f"Missing import for {imp}")

        return len(issues) == 0, issues

    def validate_class_structure(self, code):
        """
        Validate class structure

        Args:
            code (str): Python code to validate

        Returns:
            tuple: (is_valid, issues)
        """
        issues = []

        try:
            # Parse the code
            tree = ast.parse(code)

            # Find class definitions
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

            if not classes:
                issues.append("No class definition found")
                return False, issues

            # Check for required methods in the first class
            class_node = classes[0]
            method_names = [node.name for node in class_node.body if isinstance(node, ast.FunctionDef)]

            # Required methods for a trading strategy
            required_methods = ["__init__"]

            # Check if at least one signal generation method exists
            signal_methods = ["generate_signals", "backtest", "get_signals", "calculate_signals"]
            has_signal_method = any(method in method_names for method in signal_methods)

            if not has_signal_method:
                issues.append("No signal generation method found (e.g., generate_signals, backtest)")

            # Check for required methods
            for method in required_methods:
                if method not in method_names:
                    issues.append(f"Missing required method: {method}")

            return len(issues) == 0, issues

        except Exception as e:
            issues.append(f"Error validating class structure: {e}")
            return False, issues

    def validate_execution(self, code):
        """
        Validate code execution

        Args:
            code (str): Python code to validate

        Returns:
            tuple: (is_valid, issues)
        """
        issues = []

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(code)

        try:
            # Try to import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location("temp_strategy", temp_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the strategy class
            class_names = [name for name, obj in module.__dict__.items()
                           if
                           isinstance(obj, type) and hasattr(obj, '__module__') and obj.__module__ == 'temp_strategy']

            if not class_names:
                issues.append("Could not find a strategy class in the module")
                return False, issues

            # Create an instance of the class
            class_name = class_names[0]
            strategy_class = getattr(module, class_name)

            try:
                strategy = strategy_class()
            except Exception as e:
                issues.append(f"Error instantiating the strategy class: {e}")
                return False, issues

            # If we have test data, try to use generate_signals or backtest
            if self.test_data is not None:
                try:
                    if hasattr(strategy, 'generate_signals') and callable(getattr(strategy, 'generate_signals')):
                        strategy.generate_signals(self.test_data.copy())
                    elif hasattr(strategy, 'backtest') and callable(getattr(strategy, 'backtest')):
                        strategy.backtest(self.test_data.copy())
                    else:
                        issues.append("Strategy has no callable generate_signals or backtest method")
                except Exception as e:
                    issues.append(f"Error calling strategy method with test data: {e}")

            return len(issues) == 0, issues

        except Exception as e:
            issues.append(f"Error importing strategy module: {e}")
            return False, issues

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def validate(self, code):
        """
        Validate strategy code

        Args:
            code (str): Python code to validate

        Returns:
            dict: Validation results
        """
        self.logger.info("Validating strategy code")

        # Validation steps
        syntax_valid, syntax_issues = self.validate_syntax(code)

        # If syntax is invalid, no need to check further
        if not syntax_valid:
            return {
                "is_valid": False,
                "issues": syntax_issues,
                "details": {
                    "syntax": {"valid": syntax_valid, "issues": syntax_issues},
                    "imports": {"valid": False, "issues": []},
                    "class_structure": {"valid": False, "issues": []},
                    "execution": {"valid": False, "issues": []}
                }
            }

        imports_valid, import_issues = self.validate_imports(code)
        class_valid, class_issues = self.validate_class_structure(code)

        # Only run execution validation if other checks pass
        if syntax_valid and class_valid:
            execution_valid, execution_issues = self.validate_execution(code)
        else:
            execution_valid, execution_issues = False, [
                "Skipped execution validation due to syntax or class structure issues"]

        # Combine all issues
        all_issues = syntax_issues + import_issues + class_issues + execution_issues

        # Determine overall validity
        is_valid = syntax_valid and imports_valid and class_valid and execution_valid

        return {
            "is_valid": is_valid,
            "issues": all_issues,
            "details": {
                "syntax": {"valid": syntax_valid, "issues": syntax_issues},
                "imports": {"valid": imports_valid, "issues": import_issues},
                "class_structure": {"valid": class_valid, "issues": class_issues},
                "execution": {"valid": execution_valid, "issues": execution_issues}
            }
        }

    def validate_mathematical_formula(self, formula):
        """
        Validate a mathematical formula implementation

        Args:
            formula (dict): Formula dictionary with latex and python_equivalent fields

        Returns:
            dict: Validation results
        """
        if not formula.get("python_equivalent"):
            return {"is_valid": False, "issues": ["No Python implementation provided"]}

        # Try to parse the Python code
        try:
            ast.parse(formula["python_equivalent"])
            return {"is_valid": True, "issues": []}
        except SyntaxError as e:
            return {"is_valid": False, "issues": [f"Syntax error in formula: {e}"]}

    def validate_formulas(self, formulas):
        """
        Validate all formulas in a strategy

        Args:
            formulas (list): List of formula dictionaries

        Returns:
            dict: Validation results
        """
        results = []
        is_valid = True

        for i, formula in enumerate(formulas):
            validation = self.validate_mathematical_formula(formula)
            results.append({
                "formula_index": i,
                "description": formula.get("description", ""),
                "is_valid": validation["is_valid"],
                "issues": validation["issues"]
            })

            if not validation["is_valid"]:
                is_valid = False

        return {
            "is_valid": is_valid,
            "formula_results": results
        }

    def generate_test_data(self, days=500):
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

        # Store the test data
        self.test_data = df

        return df