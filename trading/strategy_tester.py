"""
Strategy Tester module for evaluating generated trading strategies
"""

import os
import importlib.util
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
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

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Load or generate test data
        self.test_data = self._load_test_data(test_data_path)

    def _load_test_data(self, test_data_path):
        """
        Load test data or generate if not available

        Args:
            test_data_path (str): Path to test data file

        Returns:
            pandas.DataFrame: Test data
        """
        if test_data_path and os.path.exists(test_data_path):
            self.logger.info(f"Loading test data from {test_data_path}")
            return pd.read_csv(test_data_path, index_col=0, parse_dates=True)
        else:
            self.logger.info("Generating synthetic test data")
            return self._generate_test_data()

    def _generate_test_data(self, days=500):
        """
        Generate synthetic market data for testing

        Args:
            days (int): Number of days to generate

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

    def test_all_strategies(self):
        """
        Test all strategies in the directory

        Returns:
            list: Test results for all strategies
        """
        strategy_files = [f for f in os.listdir(self.strategies_dir) if f.endswith('.py')]
        self.logger.info(f"Found {len(strategy_files)} strategy files to test")

        results = []
        for strategy_file in strategy_files:
            file_path = os.path.join(self.strategies_dir, strategy_file)
            try:
                self.logger.info(f"Testing strategy: {strategy_file}")
                result = self.test_strategy(file_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error testing {strategy_file}: {e}")

        # Create summary report
        self._create_summary_report(results)

        return results

    def test_strategy(self, strategy_file):
        """
        Test a single strategy file

        Args:
            strategy_file (str): Path to strategy file

        Returns:
            dict: Test results
        """
        # Load strategy class
        class_obj, class_name = self._load_strategy_class(strategy_file)

        # Create strategy instance
        strategy = class_obj()

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
            json.dump(metrics, f, indent=2, default=str)  # Use default=str to handle non-serializable objects

        # Save plot
        fig.savefig(os.path.join(results_path, 'performance.png'))
        plt.close(fig)

        # Save backtest data sample
        backtest_data.head(100).to_csv(os.path.join(results_path, 'backtest_sample.csv'))

        return {
            'strategy_name': class_name,
            'file_path': strategy_file,
            'metrics': metrics,
            'results_path': results_path
        }

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
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the strategy class
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, '__module__') and attr.__module__ == module_name:
                # Check if it's a strategy class (inherits from BaseStrategy)
                if hasattr(attr, '__bases__') and any('BaseStrategy' in str(base) for base in attr.__bases__):
                    return attr, attr_name

        raise ValueError(f"No strategy class found in {strategy_file}")

    def _create_summary_report(self, results):
        """
        Create a summary report of all tested strategies

        Args:
            results (list): List of test results
        """
        # Check if results is empty
        if not results:
            self.logger.warning("No results to create summary report")
            return

        # Create report data
        report_data = []
        for result in results:
            metrics = result['metrics']
            report_data.append({
                'strategy_name': result['strategy_name'],
                'total_return': metrics.get('total_return', 0),
                'annual_return': metrics.get('annual_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'num_trades': metrics.get('num_trades', 0),
                'results_path': result['results_path']
            })

        # Convert to DataFrame and sort by Sharpe ratio
        report_df = pd.DataFrame(report_data)
        if not report_df.empty:
            report_df = report_df.sort_values('sharpe_ratio', ascending=False)

            # Save report
            report_df.to_csv(os.path.join(self.results_dir, 'strategy_performance_summary.csv'), index=False)

            # Create comparison chart
            self._create_comparison_chart(report_df)
        else:
            self.logger.warning("Empty DataFrame, cannot create summary report")

    def _create_comparison_chart(self, report_df):
        """
        Create a chart comparing all strategies

        Args:
            report_df (pandas.DataFrame): DataFrame with strategy metrics
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot total return
        if 'total_return' in report_df.columns:
            report_df.sort_values('total_return', ascending=False).plot(
                x='strategy_name', y='total_return', kind='bar', ax=axes[0, 0], color='green')
            axes[0, 0].set_title('Total Return')
            axes[0, 0].set_ylabel('Return')
            axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

        # Plot Sharpe ratio
        if 'sharpe_ratio' in report_df.columns:
            report_df.sort_values('sharpe_ratio', ascending=False).plot(
                x='strategy_name', y='sharpe_ratio', kind='bar', ax=axes[0, 1], color='blue')
            axes[0, 1].set_title('Sharpe Ratio')
            axes[0, 1].set_ylabel('Ratio')
            axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

        # Plot max drawdown
        if 'max_drawdown' in report_df.columns:
            report_df.sort_values('max_drawdown', ascending=True).plot(
                x='strategy_name', y='max_drawdown', kind='bar', ax=axes[1, 0], color='red')
            axes[1, 0].set_title('Maximum Drawdown')
            axes[1, 0].set_ylabel('Drawdown')
            axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')

        # Plot win rate
        if 'win_rate' in report_df.columns:
            report_df.sort_values('win_rate', ascending=False).plot(
                x='strategy_name', y='win_rate', kind='bar', ax=axes[1, 1], color='purple')
            axes[1, 1].set_title('Win Rate')
            axes[1, 1].set_ylabel('Rate')
            axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'strategy_comparison.png'))
        plt.close(fig)