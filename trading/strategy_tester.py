"""
Strategy Tester module for evaluating generated trading strategies - Fixed version
"""

import os
import importlib.util
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import numpy as np
import logging


class StrategyTester:
    """
    Test runner for evaluating generated trading strategies
    """

    def __init__(self, strategies_dir, test_data_path=None, results_dir="output/test_results"):
        """
        Initialize the StrategyTester

        Args:
            strategies_dir (str): Directory containing strategy implementations
            test_data_path (str): Path to test data file
            results_dir (str): Directory to save test results
        """
        self.strategies_dir = strategies_dir
        self.results_dir = results_dir
        self.logger = logging.getLogger(__name__)

        # Initialize the config attribute
        self.config = {}

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Load or generate test data with robust error handling
        self.test_data = self._load_test_data(test_data_path)

    def _load_test_data(self, test_data_path):
        """
        Load test data with robust error handling for date parsing issues

        Args:
            test_data_path (str): Path to test data file

        Returns:
            pandas.DataFrame: Test data
        """
        if test_data_path and os.path.exists(test_data_path):
            self.logger.info(f"Loading test data from {test_data_path}")
            try:
                # Try standard loading first
                df = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
                self.logger.info(f"Successfully loaded test data with shape: {df.shape}")
                return df
            except Exception as e:
                self.logger.warning(f"Standard test data loading failed: {e}")
                try:
                    # Try loading without date parsing
                    df = pd.read_csv(test_data_path, index_col=0)
                    # Manually convert index to datetime
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    # Drop any rows with invalid dates
                    if df.index.isna().any():
                        self.logger.warning(f"Dropping {df.index.isna().sum()} rows with invalid dates")
                        df = df.loc[~df.index.isna()]
                    self.logger.info(f"Successfully loaded data with manual date conversion")
                    return df
                except Exception as e2:
                    self.logger.error(f"Failed to load test data: {e2}")
                    self.logger.warning("Falling back to synthetic test data generation")
                    return self._generate_test_data()
        else:
            self.logger.info("Test data path not provided or file not found. Generating synthetic test data")
            return self._generate_test_data()

    def _generate_test_data(self):
        """
        Generate synthetic test data for strategy testing - FIXED VERSION
        
        Returns:
            pd.DataFrame: Synthetic price data
        """
        try:
            self.logger.info("Generating synthetic test data")
            
            # Get number of days from config, with safe conversion
            days = self.config.get('test_days', 365 * 2)  # Default to 2 years
            
            # Ensure days is a plain Python int (fixes numpy.timedelta64 issue)
            if hasattr(days, 'item'):
                days = days.item()
            elif hasattr(days, 'astype'):
                days = int(days.astype(int))
            else:
                days = int(days)
            
            # Clamp to reasonable bounds
            days = max(100, min(days, 5000))  # Between 100 days and ~13 years
            
            # FIXED: Use datetime.timedelta instead of pd.Timedelta to avoid numpy issue
            end_date = datetime(2023, 12, 31)  # Use datetime object
            start_date = end_date - timedelta(days=days)  # Use datetime.timedelta
            
            # Convert to pandas Timestamp for date_range
            start_pd = pd.Timestamp(start_date)
            end_pd = pd.Timestamp(end_date)
            
            # Generate date range (business days only)
            date_range = pd.date_range(start=start_pd, end=end_pd, freq='B')  # Business days
            
            # Limit to reasonable size if too large
            if len(date_range) > 2000:
                date_range = date_range[-2000:]  # Take last 2000 business days
            
            self.logger.info(f"Generated date range with {len(date_range)} business days")
            
            # Generate realistic price data
            np.random.seed(42)  # For reproducibility
            
            # Parameters for price generation
            initial_price = 100.0
            annual_volatility = 0.20  # 20% annual volatility
            daily_volatility = annual_volatility / np.sqrt(252)  # Convert to daily
            annual_drift = 0.05  # 5% annual drift
            daily_drift = annual_drift / 252  # Convert to daily
            
            # Generate daily returns with drift
            num_days = len(date_range)
            daily_returns = np.random.normal(daily_drift, daily_volatility, num_days)
            
            # Calculate cumulative price series using geometric returns
            cumulative_returns = np.cumprod(1 + daily_returns)
            close_prices = initial_price * cumulative_returns
            
            # Generate OHLC data with realistic relationships
            # Open prices: previous close + small gap
            open_prices = np.zeros(num_days)
            open_prices[0] = initial_price
            for i in range(1, num_days):
                gap = np.random.normal(0, 0.002)  # Small overnight gap
                open_prices[i] = close_prices[i-1] * (1 + gap)
            
            # High and low prices: realistic intraday ranges
            intraday_range = np.random.uniform(0.005, 0.025, num_days)  # 0.5% to 2.5% daily range
            
            high_prices = np.maximum(open_prices, close_prices) * (1 + intraday_range * 0.7)
            low_prices = np.minimum(open_prices, close_prices) * (1 - intraday_range * 0.7)
            
            # Ensure OHLC relationships are maintained
            high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
            low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
            
            # Generate realistic volume data
            base_volume = 1000000
            volume_noise = np.random.uniform(0.5, 2.0, num_days)
            volumes = (base_volume * volume_noise).astype(int)
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes
            }, index=date_range)
            
            # Ensure index is properly set as DatetimeIndex
            df.index.name = 'date'
            
            # Basic validation
            if len(df) == 0:
                raise ValueError("Generated empty DataFrame")
            
            # Check for any NaN values
            if df.isnull().any().any():
                self.logger.warning("Generated data contains NaN values, filling with forward fill")
                df = df.fillna(method='ffill')
            
            self.logger.info(f"Successfully generated test data with shape: {df.shape}")
            self.logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            self.logger.info(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic test data: {e}")
            return self._create_minimal_fallback_data()

    def _create_minimal_fallback_data(self):
        """
        Create minimal fallback test data when all else fails
        
        Returns:
            pd.DataFrame: Minimal test data
        """
        try:
            self.logger.warning("Creating minimal fallback test data")
            
            # Create simple dates using datetime objects
            start_date = datetime(2023, 1, 1)
            dates = [start_date + timedelta(days=i) for i in range(100)]
            
            # Convert to pandas DatetimeIndex
            date_index = pd.DatetimeIndex(dates)
            
            # Simple price data
            np.random.seed(42)
            base_price = 100.0
            price_changes = np.random.normal(0, 1, 100)
            prices = base_price + np.cumsum(price_changes)
            
            # Ensure prices are positive
            prices = np.abs(prices) + 50  # Minimum $50
            
            # Create OHLCV data
            df = pd.DataFrame({
                'open': prices * np.random.uniform(0.995, 1.005, 100),
                'high': prices * np.random.uniform(1.01, 1.03, 100),
                'low': prices * np.random.uniform(0.97, 0.99, 100),
                'close': prices,
                'volume': np.random.randint(100000, 1000000, 100)
            }, index=date_index)
            
            # Ensure OHLC relationships
            df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
            df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
            
            df.index.name = 'date'
            
            self.logger.info(f"Created minimal fallback data with {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Even minimal fallback failed: {e}")
            # Last resort: hardcoded data
            df = pd.DataFrame({
                'open': [100, 101, 102, 103, 104],
                'high': [102, 103, 104, 105, 106],
                'low': [99, 100, 101, 102, 103],
                'close': [101, 102, 103, 104, 105],
                'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
            })
            df.index.name = 'date'
            return df

    def test_all_strategies(self):
        """
        Test all strategies in the directory

        Returns:
            list: Test results for all strategies
        """
        if not os.path.exists(self.strategies_dir):
            self.logger.warning(f"Strategies directory does not exist: {self.strategies_dir}")
            return []
        
        strategy_files = [f for f in os.listdir(self.strategies_dir) if f.endswith('.py') and not f.startswith('__')]
        self.logger.info(f"Found {len(strategy_files)} strategy files to test")

        if not strategy_files:
            self.logger.warning("No strategy files found to test")
            return []

        results = []
        for strategy_file in strategy_files:
            file_path = os.path.join(self.strategies_dir, strategy_file)
            try:
                self.logger.info(f"Testing strategy: {strategy_file}")
                result = self.test_strategy(file_path)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"Error testing {strategy_file}: {e}")

        # Create summary report if we have results
        if results:
            self._create_summary_report(results)
        else:
            self.logger.warning("No successful strategy tests to create summary report")

        return results

    def test_strategy(self, strategy_file):
        """
        Test a single strategy file

        Args:
            strategy_file (str): Path to strategy file

        Returns:
            dict: Test results
        """
        try:
            # Load strategy class
            class_obj, class_name = self._load_strategy_class(strategy_file)

            # Create strategy instance
            strategy = class_obj()

            # Ensure we have test data
            if self.test_data is None or len(self.test_data) == 0:
                self.logger.error("No test data available for strategy testing")
                return None

            # Run backtest
            backtest_data, metrics = strategy.backtest(self.test_data.copy())

            # Create visualizations
            fig = strategy.plot_results(backtest_data)

            # Save results
            base_name = os.path.basename(strategy_file).replace('.py', '')
            results_path = os.path.join(self.results_dir, base_name)
            os.makedirs(results_path, exist_ok=True)

            # Save metrics
            with open(os.path.join(results_path, 'metrics.json'), 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                serializable_metrics = {}
                for key, value in metrics.items():
                    if hasattr(value, 'item'):
                        serializable_metrics[key] = value.item()
                    elif isinstance(value, (np.integer, np.floating)):
                        serializable_metrics[key] = float(value)
                    else:
                        serializable_metrics[key] = value
                json.dump(serializable_metrics, f, indent=2)

            # Save plot
            fig.savefig(os.path.join(results_path, 'performance.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Save backtest data sample
            backtest_sample = backtest_data.head(min(100, len(backtest_data)))
            backtest_sample.to_csv(os.path.join(results_path, 'backtest_sample.csv'))

            self.logger.info(f"Successfully tested strategy {class_name}")

            return {
                'strategy_name': class_name,
                'file_path': strategy_file,
                'metrics': serializable_metrics,
                'results_path': results_path
            }

        except Exception as e:
            self.logger.error(f"Error testing strategy {strategy_file}: {e}")
            return None

    def _load_strategy_class(self, strategy_file):
        """
        Load a strategy class from a file

        Args:
            strategy_file (str): Path to strategy file

        Returns:
            tuple: (class_object, class_name)
        """
        # Import the module
        module_name = os.path.basename(strategy_file).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, strategy_file)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load module from {strategy_file}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the strategy class
        strategy_classes = []
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, '__module__') and attr.__module__ == module_name:
                # Check if it's a strategy class (has required methods)
                if hasattr(attr, 'backtest') or hasattr(attr, 'generate_signals'):
                    strategy_classes.append((attr, attr_name))

        if not strategy_classes:
            raise ValueError(f"No strategy class found in {strategy_file}")
        
        # Return the first strategy class found
        return strategy_classes[0]

    def _create_summary_report(self, results):
        """
        Create a summary report of all tested strategies

        Args:
            results (list): List of test results
        """
        if not results:
            self.logger.warning("No results to create summary report")
            return

        try:
            # Create report data
            report_data = []
            for result in results:
                metrics = result.get('metrics', {})
                report_data.append({
                    'strategy_name': result.get('strategy_name', 'Unknown'),
                    'total_return': metrics.get('total_return', 0),
                    'annual_return': metrics.get('annual_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'num_trades': metrics.get('num_trades', 0),
                    'results_path': result.get('results_path', '')
                })

            # Convert to DataFrame
            report_df = pd.DataFrame(report_data)
            
            if not report_df.empty:
                # Sort by Sharpe ratio
                report_df = report_df.sort_values('sharpe_ratio', ascending=False)

                # Save report
                report_path = os.path.join(self.results_dir, 'strategy_performance_summary.csv')
                report_df.to_csv(report_path, index=False)
                self.logger.info(f"Saved strategy performance summary to {report_path}")

                # Create comparison chart
                self._create_comparison_chart(report_df)
            else:
                self.logger.warning("Empty DataFrame, cannot create summary report")

        except Exception as e:
            self.logger.error(f"Error creating summary report: {e}")

    def _create_comparison_chart(self, report_df):
        """
        Create a chart comparing all strategies

        Args:
            report_df (pandas.DataFrame): DataFrame with strategy metrics
        """
        try:
            if len(report_df) == 0:
                self.logger.warning("No data to create comparison chart")
                return

            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Strategy Performance Comparison', fontsize=16)

            # Plot total return
            if 'total_return' in report_df.columns and not report_df['total_return'].isna().all():
                sorted_df = report_df.sort_values('total_return', ascending=False)
                axes[0, 0].bar(range(len(sorted_df)), sorted_df['total_return'], color='green', alpha=0.7)
                axes[0, 0].set_title('Total Return')
                axes[0, 0].set_ylabel('Return')
                axes[0, 0].set_xticks(range(len(sorted_df)))
                axes[0, 0].set_xticklabels(sorted_df['strategy_name'], rotation=45, ha='right')

            # Plot Sharpe ratio
            if 'sharpe_ratio' in report_df.columns and not report_df['sharpe_ratio'].isna().all():
                sorted_df = report_df.sort_values('sharpe_ratio', ascending=False)
                axes[0, 1].bar(range(len(sorted_df)), sorted_df['sharpe_ratio'], color='blue', alpha=0.7)
                axes[0, 1].set_title('Sharpe Ratio')
                axes[0, 1].set_ylabel('Ratio')
                axes[0, 1].set_xticks(range(len(sorted_df)))
                axes[0, 1].set_xticklabels(sorted_df['strategy_name'], rotation=45, ha='right')

            # Plot max drawdown
            if 'max_drawdown' in report_df.columns and not report_df['max_drawdown'].isna().all():
                sorted_df = report_df.sort_values('max_drawdown', ascending=True)
                axes[1, 0].bar(range(len(sorted_df)), sorted_df['max_drawdown'], color='red', alpha=0.7)
                axes[1, 0].set_title('Maximum Drawdown')
                axes[1, 0].set_ylabel('Drawdown')
                axes[1, 0].set_xticks(range(len(sorted_df)))
                axes[1, 0].set_xticklabels(sorted_df['strategy_name'], rotation=45, ha='right')

            # Plot win rate
            if 'win_rate' in report_df.columns and not report_df['win_rate'].isna().all():
                sorted_df = report_df.sort_values('win_rate', ascending=False)
                axes[1, 1].bar(range(len(sorted_df)), sorted_df['win_rate'], color='purple', alpha=0.7)
                axes[1, 1].set_title('Win Rate')
                axes[1, 1].set_ylabel('Rate')
                axes[1, 1].set_xticks(range(len(sorted_df)))
                axes[1, 1].set_xticklabels(sorted_df['strategy_name'], rotation=45, ha='right')

            plt.tight_layout()
            chart_path = os.path.join(self.results_dir, 'strategy_comparison.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Saved strategy comparison chart to {chart_path}")

        except Exception as e:
            self.logger.error(f"Error creating comparison chart: {e}")
            plt.close('all')  # Clean up any open figures