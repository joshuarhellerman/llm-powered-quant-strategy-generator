"""
Enhanced Validation Service - Integrated with refactored pipeline
Combines the bulletproof approach of your existing service with BaseStrategy compliance
"""

import os
import re
import ast
import logging
import tempfile
import pandas as pd
import numpy as np
import signal
import datetime
import importlib.util
import sys
from typing import Dict, List, Any, Optional


class ValidationService:
    """
    Enhanced service for validating trading strategy implementations
    Combines bulletproof data generation with BaseStrategy compliance checking
    """

    def __init__(self, config=None):
        """
        Initialize the ValidationService

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize test data as None
        self.test_data = None

        # Try to load existing test data if available
        test_data_path = self.config.get("test_data_path")
        if test_data_path and os.path.exists(test_data_path):
            self._try_load_existing_test_data(test_data_path)

    def _try_load_existing_test_data(self, test_data_path):
        """Safely try to load existing test data"""
        try:
            self.logger.info(f"Attempting to load existing test data from {test_data_path}")
            # Try multiple loading methods
            try:
                # Method 1: Standard loading
                self.test_data = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
                self.logger.info(f"Successfully loaded test data with shape {self.test_data.shape}")
                return
            except Exception as e1:
                self.logger.debug(f"Standard loading failed: {e1}")
                # Method 2: Load without date parsing
                self.test_data = pd.read_csv(test_data_path, index_col=0)
                # Try to convert index to datetime manually
                try:
                    self.test_data.index = pd.to_datetime(self.test_data.index, errors='coerce')
                    self.logger.info(f"Loaded test data with manual date conversion, shape {self.test_data.shape}")
                    return
                except Exception as e2:
                    self.logger.debug(f"Date conversion failed: {e2}")
                    # Keep the data as-is without datetime index
                    self.logger.info(f"Loaded test data without date conversion, shape {self.test_data.shape}")
                    return

        except Exception as e:
            self.logger.warning(f"Could not load existing test data: {e}")
            self.test_data = None

    def validate_syntax(self, code):
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            return False, [f"Syntax error at line {e.lineno}: {e.msg}"]
        except Exception as e:
            return False, [f"Error parsing code: {e}"]

    def validate_imports(self, code):
        """Enhanced import validation for BaseStrategy compliance"""
        issues = []

        # Required imports for BaseStrategy compliance
        required_imports = [
            ("pandas", r'\bimport\s+pandas\b|\bfrom\s+pandas\b'),
            ("numpy", r'\bimport\s+numpy\b|\bfrom\s+numpy\b'),
            ("BaseStrategy", r'\bfrom\s+base_strategy\s+import\s+BaseStrategy\b|\bfrom\s+.*base_strategy.*\s+import\s+.*BaseStrategy')
        ]

        for import_name, pattern in required_imports:
            if not re.search(pattern, code):
                if import_name == "BaseStrategy":
                    issues.append("Missing required import: 'from base_strategy import BaseStrategy'")
                else:
                    issues.append(f"Missing required import for {import_name}")

        return len(issues) == 0, issues

    def validate_class_structure(self, code):
        """Enhanced class structure validation for BaseStrategy compliance"""
        issues = []

        try:
            tree = ast.parse(code)
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

            if not classes:
                issues.append("No class definition found")
                return False, issues

            # Check for BaseStrategy inheritance
            class_node = classes[0]

            # Check if class inherits from BaseStrategy
            inherits_from_base = False
            if class_node.bases:
                for base in class_node.bases:
                    if isinstance(base, ast.Name) and base.id == 'BaseStrategy':
                        inherits_from_base = True
                        break

            if not inherits_from_base:
                issues.append("Class must inherit from BaseStrategy: 'class YourStrategy(BaseStrategy):'")

            # Check for required methods
            method_names = [node.name for node in class_node.body if isinstance(node, ast.FunctionDef)]

            # Required abstract methods for BaseStrategy
            required_methods = ["_initialize_parameters", "generate_signals"]

            for method in required_methods:
                if method not in method_names:
                    issues.append(f"Missing required abstract method: {method}")

            # Check for forbidden methods (should be inherited from BaseStrategy)
            forbidden_methods = ["backtest", "plot_results", "_calculate_metrics"]
            for method in forbidden_methods:
                if method in method_names:
                    issues.append(f"Do not implement '{method}' method - it's provided by BaseStrategy")

            # Validate method signatures
            for node in class_node.body:
                if isinstance(node, ast.FunctionDef):
                    if node.name == "_initialize_parameters":
                        # Check signature: def _initialize_parameters(self, params):
                        if len(node.args.args) != 2:  # self + params
                            issues.append("_initialize_parameters must have signature: def _initialize_parameters(self, params)")
                    elif node.name == "generate_signals":
                        # Check signature: def generate_signals(self, data):
                        if len(node.args.args) != 2:  # self + data
                            issues.append("generate_signals must have signature: def generate_signals(self, data)")

            return len(issues) == 0, issues

        except Exception as e:
            issues.append(f"Error validating class structure: {e}")
            return False, issues

    def validate_signal_generation_logic(self, code):
        """Validate signal generation logic requirements"""
        issues = []

        # Check for signal column initialization
        if "df['signal'] = 0" not in code and "signal" in code:
            issues.append("Must initialize signal column with: df['signal'] = 0")

        # Check for proper signal values documentation or usage
        signal_patterns = [
            r"signal.*=.*1",  # Buy signals
            r"signal.*=.*-1", # Sell signals
            r"signal.*=.*0"   # Hold signals
        ]

        has_signal_logic = any(re.search(pattern, code) for pattern in signal_patterns)
        if not has_signal_logic:
            issues.append("No signal generation logic found (should use 1=buy, -1=sell, 0=hold)")

        return len(issues) == 0, issues

    def validate_execution_with_base_strategy(self, code):
        """Enhanced execution validation that tests BaseStrategy compliance"""
        issues = []

        def timeout_handler(signum, frame):
            raise TimeoutError("Strategy execution timed out")

        # Ensure we have test data
        if self.test_data is None:
            self.test_data = self.generate_test_data()

        # Create a temporary file with proper imports
        enhanced_code = f"""
import sys
import os

# Add strategy_format to path for BaseStrategy import
current_dir = os.path.dirname(os.path.abspath(__file__))
strategy_format_dir = os.path.join(current_dir, '..', '..', '..', 'strategy_format')
if os.path.exists(strategy_format_dir) and strategy_format_dir not in sys.path:
    sys.path.insert(0, strategy_format_dir)

# Also try common locations for BaseStrategy
possible_paths = [
    os.path.join(current_dir, 'strategy_format'),
    os.path.join(current_dir, '..', 'strategy_format'),
    os.path.join(current_dir, '..', '..', 'strategy_format'),
    'strategy_format'
]

for path in possible_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

{code}
"""

        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(enhanced_code)

        try:
            # Set a timeout for execution (15 seconds for BaseStrategy testing)
            timeout_set = False
            if hasattr(signal, 'SIGALRM') and os.name != 'nt':  # Not on Windows
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(15)  # Longer timeout for BaseStrategy testing
                    timeout_set = True
                except Exception:
                    self.logger.debug("Could not set timeout - continuing without it")

            # Try to import the module
            spec = importlib.util.spec_from_file_location("temp_strategy", temp_file_path)
            if spec is None or spec.loader is None:
                issues.append("Could not create module spec from strategy file")
                return False, issues

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the strategy class
            strategy_classes = []
            for name, obj in module.__dict__.items():
                if (isinstance(obj, type) and
                    hasattr(obj, '__module__') and
                    obj.__module__ == 'temp_strategy' and
                    name != 'BaseStrategy'):
                    strategy_classes.append((name, obj))

            if not strategy_classes:
                issues.append("Could not find a strategy class in the module")
                return False, issues

            # Test the first strategy class
            class_name, strategy_class = strategy_classes[0]

            # Check if it's actually a subclass of BaseStrategy
            try:
                # Try to find BaseStrategy in the module or imported modules
                base_strategy_class = None
                if hasattr(module, 'BaseStrategy'):
                    base_strategy_class = module.BaseStrategy
                else:
                    # Try to import BaseStrategy directly
                    try:
                        from base_strategy import BaseStrategy as ImportedBaseStrategy
                        base_strategy_class = ImportedBaseStrategy
                    except ImportError:
                        pass

                if base_strategy_class and not issubclass(strategy_class, base_strategy_class):
                    issues.append(f"Class {class_name} does not inherit from BaseStrategy")

            except Exception as e:
                self.logger.warning(f"Could not verify BaseStrategy inheritance: {e}")

            # Try to instantiate the strategy
            try:
                strategy = strategy_class()
                self.logger.info(f"Successfully instantiated strategy class: {class_name}")
            except TypeError as e:
                if "abstract methods" in str(e):
                    # Extract method names from error message
                    import re
                    methods = re.findall(r"'(\w+)'", str(e))
                    issues.append(f"Abstract methods not implemented: {', '.join(methods)}")
                else:
                    issues.append(f"Error instantiating strategy class: {e}")
                return False, issues
            except Exception as e:
                issues.append(f"Error instantiating strategy class: {e}")
                return False, issues

            # Test required methods exist and are callable
            required_methods = ['_initialize_parameters', 'generate_signals']
            for method_name in required_methods:
                if not hasattr(strategy, method_name):
                    issues.append(f"Strategy missing required method: {method_name}")
                elif not callable(getattr(strategy, method_name)):
                    issues.append(f"Strategy method {method_name} is not callable")

            # Test generate_signals with real data
            try:
                test_data_copy = self.test_data.copy()
                self.logger.info(f"Testing generate_signals with data shape: {test_data_copy.shape}")

                result = strategy.generate_signals(test_data_copy)

                if result is None:
                    issues.append("generate_signals method returned None")
                elif not isinstance(result, pd.DataFrame):
                    issues.append(f"generate_signals must return DataFrame, got {type(result)}")
                elif 'signal' not in result.columns:
                    issues.append("generate_signals must return DataFrame with 'signal' column")
                else:
                    # Validate signal values
                    unique_signals = result['signal'].unique()
                    valid_signals = [-1, 0, 1]
                    invalid_signals = [s for s in unique_signals if s not in valid_signals and not pd.isna(s)]
                    if invalid_signals:
                        issues.append(f"Invalid signal values found: {invalid_signals}. Must use -1, 0, 1")

                    self.logger.info(f"generate_signals test passed. Unique signals: {unique_signals}")

            except Exception as e:
                issues.append(f"Error testing generate_signals method: {e}")
                self.logger.error(f"generate_signals test failed: {e}")

            # Test backtest method (should be inherited from BaseStrategy)
            try:
                test_data_copy = self.test_data.copy()
                backtest_result = strategy.backtest(test_data_copy)

                if backtest_result is None:
                    issues.append("backtest method returned None")
                elif not isinstance(backtest_result, tuple) or len(backtest_result) != 2:
                    issues.append("backtest must return tuple of (DataFrame, dict)")
                else:
                    backtest_data, metrics = backtest_result
                    if not isinstance(backtest_data, pd.DataFrame):
                        issues.append("backtest must return DataFrame as first element")
                    if not isinstance(metrics, dict):
                        issues.append("backtest must return dict as second element")
                    else:
                        # Check for required metrics
                        required_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
                        for metric in required_metrics:
                            if metric not in metrics:
                                issues.append(f"Missing required metric in backtest results: {metric}")

                        self.logger.info(f"backtest test passed. Metrics: {list(metrics.keys())}")

            except Exception as e:
                issues.append(f"Error testing backtest method: {e}")
                self.logger.error(f"backtest test failed: {e}")

            return len(issues) == 0, issues

        except TimeoutError:
            issues.append("Strategy execution timed out (>15 seconds)")
            return False, issues
        except Exception as e:
            issues.append(f"Error during strategy execution test: {e}")
            self.logger.error(f"Execution test error: {e}")
            return False, issues

        finally:
            # Cancel the alarm if it was set
            if timeout_set and hasattr(signal, 'SIGALRM'):
                try:
                    signal.alarm(0)
                except Exception:
                    pass
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

    def validate(self, code):
        """Comprehensive validation with BaseStrategy compliance"""
        self.logger.info("Starting comprehensive BaseStrategy validation")

        # Step 1: Syntax validation
        syntax_valid, syntax_issues = self.validate_syntax(code)
        if not syntax_valid:
            return {
                "is_valid": False,
                "issues": syntax_issues,
                "details": {
                    "syntax": {"valid": syntax_valid, "issues": syntax_issues},
                    "imports": {"valid": False, "issues": []},
                    "class_structure": {"valid": False, "issues": []},
                    "signal_logic": {"valid": False, "issues": []},
                    "execution": {"valid": False, "issues": []}
                }
            }

        # Step 2: Import validation
        imports_valid, import_issues = self.validate_imports(code)

        # Step 3: Class structure validation
        class_valid, class_issues = self.validate_class_structure(code)

        # Step 4: Signal generation logic validation
        signal_logic_valid, signal_logic_issues = self.validate_signal_generation_logic(code)

        # Step 5: Execution validation (only if previous checks pass)
        if syntax_valid and imports_valid and class_valid:
            execution_valid, execution_issues = self.validate_execution_with_base_strategy(code)
        else:
            execution_valid, execution_issues = False, [
                "Skipped execution validation due to syntax, import, or class structure issues"
            ]

        # Combine all issues
        all_issues = syntax_issues + import_issues + class_issues + signal_logic_issues + execution_issues

        # Determine overall validity
        is_valid = (syntax_valid and imports_valid and class_valid and
                   signal_logic_valid and execution_valid)

        result = {
            "is_valid": is_valid,
            "issues": all_issues,
            "details": {
                "syntax": {"valid": syntax_valid, "issues": syntax_issues},
                "imports": {"valid": imports_valid, "issues": import_issues},
                "class_structure": {"valid": class_valid, "issues": class_issues},
                "signal_logic": {"valid": signal_logic_valid, "issues": signal_logic_issues},
                "execution": {"valid": execution_valid, "issues": execution_issues}
            }
        }

        if is_valid:
            self.logger.info("✅ Strategy validation passed - BaseStrategy compliant")
        else:
            self.logger.warning(f"❌ Strategy validation failed with {len(all_issues)} issues")
            for issue in all_issues:
                self.logger.warning(f"  - {issue}")

        return result

    def validate_mathematical_formula(self, formula):
        """Validate a mathematical formula implementation"""
        if not formula.get("python_equivalent"):
            return {"is_valid": False, "issues": ["No Python implementation provided"]}

        # Try to parse the Python code
        try:
            ast.parse(formula["python_equivalent"])
            return {"is_valid": True, "issues": []}
        except SyntaxError as e:
            return {"is_valid": False, "issues": [f"Syntax error in formula: {e}"]}
        except Exception as e:
            return {"is_valid": False, "issues": [f"Error parsing formula: {e}"]}

    def validate_formulas(self, formulas):
        """Validate all formulas in a strategy"""
        results = []
        is_valid = True

        if not formulas:
            return {"is_valid": True, "formula_results": []}

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

    def generate_test_data(self, start_date="2020-01-01", end_date="2023-12-31", num_days=500):
        """
        Generate synthetic market data for testing strategies - BULLETPROOF VERSION
        Uses your proven hardcoded approach
        """
        self.logger.info(f"Generating test data (bulletproof mode)")

        # Use your proven hardcoded approach
        try:
            return self._create_hardcoded_test_data()
        except Exception as e:
            self.logger.error(f"Even hardcoded test data failed: {e}")
            # Return absolute minimum data structure
            return pd.DataFrame({
                'open': [100, 101, 102, 103, 104],
                'high': [102, 103, 104, 105, 106],
                'low': [98, 99, 100, 101, 102],
                'close': [101, 102, 103, 104, 105],
                'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
            })

    def _create_hardcoded_test_data(self):
        """
        Create hardcoded test data that's guaranteed to work
        Uses your proven implementation
        """
        self.logger.info("Creating hardcoded test data")

        # Use your proven hardcoded data
        data = {
            'open': [100.0, 101.0, 99.5, 102.0, 101.5, 103.0, 102.0, 104.5, 103.0, 105.0,
                     104.0, 106.5, 105.0, 107.0, 106.0, 108.5, 107.0, 109.0, 108.0, 110.5,
                     109.0, 111.0, 110.0, 112.5, 111.0, 113.0, 112.0, 114.5, 113.0, 115.0,
                     114.0, 116.5, 115.0, 117.0, 116.0, 118.5, 117.0, 119.0, 118.0, 120.5,
                     119.0, 121.0, 120.0, 122.5, 121.0, 123.0, 122.0, 124.5, 123.0, 125.0],

            'high': [102.0, 103.0, 101.5, 104.0, 103.5, 105.0, 104.0, 106.5, 105.0, 107.0,
                     106.0, 108.5, 107.0, 109.0, 108.0, 110.5, 109.0, 111.0, 110.0, 112.5,
                     111.0, 113.0, 112.0, 114.5, 113.0, 115.0, 114.0, 116.5, 115.0, 117.0,
                     116.0, 118.5, 117.0, 119.0, 118.0, 120.5, 119.0, 121.0, 120.0, 122.5,
                     121.0, 123.0, 122.0, 124.5, 123.0, 125.0, 124.0, 126.5, 125.0, 127.0],

            'low': [99.0, 100.0, 98.5, 101.0, 100.5, 102.0, 101.0, 103.5, 102.0, 104.0,
                    103.0, 105.5, 104.0, 106.0, 105.0, 107.5, 106.0, 108.0, 107.0, 109.5,
                    108.0, 110.0, 109.0, 111.5, 110.0, 112.0, 111.0, 113.5, 112.0, 114.0,
                    113.0, 115.5, 114.0, 116.0, 115.0, 117.5, 116.0, 118.0, 117.0, 119.5,
                    118.0, 120.0, 119.0, 121.5, 120.0, 122.0, 121.0, 123.5, 122.0, 124.0],

            'close': [101.0, 102.0, 100.5, 103.0, 102.5, 104.0, 103.0, 105.5, 104.0, 106.0,
                      105.0, 107.5, 106.0, 108.0, 107.0, 109.5, 108.0, 110.0, 109.0, 111.5,
                      110.0, 112.0, 111.0, 113.5, 112.0, 114.0, 113.0, 115.5, 114.0, 116.0,
                      115.0, 117.5, 116.0, 118.0, 117.0, 119.5, 118.0, 120.0, 119.0, 121.5,
                      120.0, 122.0, 121.0, 123.5, 122.0, 124.0, 123.0, 125.5, 124.0, 126.0],

            'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000,
                       2000000, 2100000, 2200000, 2300000, 2400000, 2500000, 2600000, 2700000, 2800000, 2900000,
                       3000000, 3100000, 3200000, 3300000, 3400000, 3500000, 3600000, 3700000, 3800000, 3900000,
                       4000000, 4100000, 4200000, 4300000, 4400000, 4500000, 4600000, 4700000, 4800000, 4900000,
                       5000000, 5100000, 5200000, 5300000, 5400000, 5500000, 5600000, 5700000, 5800000, 5900000]
        }

        # Create DataFrame with simple integer index (no dates to cause issues)
        df = pd.DataFrame(data)

        # Store the test data
        self.test_data = df

        self.logger.info(f"Created hardcoded test data with {len(df)} rows")

        # Try to save to file if path specified (but don't fail if this doesn't work)
        test_data_path = self.config.get("test_data_path")
        if test_data_path:
            try:
                os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
                df.to_csv(test_data_path, index=False)
                self.logger.info(f"Saved hardcoded test data to {test_data_path}")
            except Exception as save_error:
                self.logger.warning(f"Could not save test data (continuing anyway): {save_error}")

        return df