"""
Enhanced BaseStrategy class with improved validation and error handling
This replaces the existing strategy_format/base_strategy.py file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import logging


class BaseStrategy(ABC):
    """
    Enhanced base class for all trading strategies.
    All LLM-generated strategies should inherit from this class.

    This class provides:
    - Standardized interface for all trading strategies
    - Built-in backtesting functionality
    - Performance metrics calculation
    - Result visualization
    - Input validation and error handling
    """

    def __init__(self, **kwargs):
        """
        Initialize the strategy with parameters

        Args:
            **kwargs: Strategy parameters
        """
        # Standard parameters all strategies should have
        self.name = kwargs.get('name', self.__class__.__name__)
        self.description = kwargs.get('description', '')
        self.parameters = kwargs

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize additional parameters
        try:
            self._initialize_parameters(kwargs)
            self.logger.info(f"Strategy {self.name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing strategy parameters: {e}")
            raise

    @abstractmethod
    def _initialize_parameters(self, params):
        """
        Initialize strategy-specific parameters

        Args:
            params (dict): Parameter dictionary

        This method must be implemented by all strategy subclasses.
        Use this to set up all strategy-specific parameters.

        Example:
            def _initialize_parameters(self, params):
                self.fast_window = params.get('fast_window', 10)
                self.slow_window = params.get('slow_window', 50)
                self.threshold = params.get('threshold', 0.02)
        """
        pass

    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals for the given data

        Args:
            data (pd.DataFrame): Market data with OHLCV columns
                Required columns: ['open', 'high', 'low', 'close', 'volume']

        Returns:
            pd.DataFrame: Data with added signal column (1=buy, -1=sell, 0=hold)

        This method must be implemented by all strategy subclasses.
        The returned DataFrame must include a 'signal' column with:
        - 1 for buy signals
        - -1 for sell signals
        - 0 for hold/no action

        Example:
            def generate_signals(self, data):
                df = data.copy()
                df['signal'] = 0  # Initialize signals

                # Calculate indicators
                df['ma_fast'] = df['close'].rolling(self.fast_window).mean()
                df['ma_slow'] = df['close'].rolling(self.slow_window).mean()

                # Generate signals
                df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
                df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1

                return df
        """
        pass

    def validate_data(self, data):
        """
        Validate input data format and content

        Args:
            data (pd.DataFrame): Input market data

        Returns:
            pd.DataFrame: Validated and cleaned data

        Raises:
            ValueError: If data doesn't meet requirements
        """
        if data is None or len(data) == 0:
            raise ValueError("Data is empty or None")

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data must be a pandas DataFrame, got {type(data)}")

        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")

        # Check for numeric data
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except:
                    raise ValueError(f"Column {col} cannot be converted to numeric")

        # Check for OHLC relationships
        ohlc_issues = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )

        if ohlc_issues.any():
            self.logger.warning(f"Found {ohlc_issues.sum()} rows with invalid OHLC relationships")
            # Fix invalid relationships
            data.loc[ohlc_issues, 'high'] = data.loc[ohlc_issues, ['open', 'close']].max(axis=1)
            data.loc[ohlc_issues, 'low'] = data.loc[ohlc_issues, ['open', 'close']].min(axis=1)

        # Check for negative volumes
        negative_volume = data['volume'] < 0
        if negative_volume.any():
            self.logger.warning(f"Found {negative_volume.sum()} rows with negative volume, setting to 0")
            data.loc[negative_volume, 'volume'] = 0

        # Check for NaN values
        nan_counts = data[required_columns].isnull().sum()
        if nan_counts.any():
            self.logger.warning(f"Found NaN values: {nan_counts.to_dict()}")
            # Forward fill NaN values
            data[required_columns] = data[required_columns].fillna(method='ffill')
            # If still NaN (at beginning), backward fill
            data[required_columns] = data[required_columns].fillna(method='bfill')

        return data

    def preprocess_data(self, data):
        """
        Preprocess data before generating signals

        Args:
            data (pd.DataFrame): Raw market data

        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Validate input data
        data = self.validate_data(data)

        # Ensure column names are lowercase
        data.columns = [col.lower() for col in data.columns]

        # Sort by index if it's a DatetimeIndex
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()

        # Remove any duplicate index entries
        if data.index.duplicated().any():
            self.logger.warning("Found duplicate index entries, keeping last occurrence")
            data = data[~data.index.duplicated(keep='last')]

        return data

    def validate_signals(self, data):
        """
        Validate generated signals

        Args:
            data (pd.DataFrame): Data with signals

        Returns:
            pd.DataFrame: Data with validated signals
        """
        if 'signal' not in data.columns:
            raise ValueError("Signal column not found in data")

        # Check signal values are valid
        valid_signals = [-1, 0, 1]
        invalid_signals = ~data['signal'].isin(valid_signals)

        if invalid_signals.any():
            self.logger.warning(f"Found {invalid_signals.sum()} invalid signal values, setting to 0")
            data.loc[invalid_signals, 'signal'] = 0

        # Fill any NaN signal values
        if data['signal'].isnull().any():
            self.logger.warning("Found NaN signal values, filling with 0")
            data['signal'] = data['signal'].fillna(0)

        # Ensure signals are integers
        data['signal'] = data['signal'].astype(int)

        return data

    def backtest(self, data):
        """
        Backtest the strategy on historical data

        Args:
            data (pd.DataFrame): Market data with OHLCV columns

        Returns:
            tuple: (backtest_data, metrics)
        """
        try:
            # Preprocess data
            data = self.preprocess_data(data.copy())

            # Check if we have enough data
            if len(data) < 2:
                raise ValueError("Insufficient data for backtesting (need at least 2 rows)")

            # Generate signals
            self.logger.info(f"Generating signals for {len(data)} data points")
            data = self.generate_signals(data)

            # Validate signals
            data = self.validate_signals(data)

            # Calculate positions (signals shifted by 1 to avoid look-ahead bias)
            data['position'] = data['signal'].shift(1).fillna(0)

            # Calculate market returns
            data['market_returns'] = data['close'].pct_change().fillna(0)

            # Calculate strategy returns
            data['strategy_returns'] = data['position'] * data['market_returns']

            # Calculate cumulative returns
            data['cumulative_market_returns'] = (1 + data['market_returns']).cumprod() - 1
            data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod() - 1

            # Calculate performance metrics
            metrics = self._calculate_metrics(data)

            self.logger.info(f"Backtest completed. Total return: {metrics['total_return']:.2%}")

            return data, metrics

        except Exception as e:
            self.logger.error(f"Error during backtesting: {e}")
            raise

    def _calculate_metrics(self, data):
        """Calculate performance metrics"""
        try:
            # Basic metrics
            total_return = data['cumulative_strategy_returns'].iloc[-1]

            # Handle case where we have no data
            if len(data) == 0:
                return self._get_zero_metrics()

            # Annualized return and volatility
            trading_days = 252
            years = len(data) / trading_days

            if years > 0:
                annual_return = (1 + total_return) ** (1/years) - 1
            else:
                annual_return = 0

            # Calculate volatility
            strategy_returns = data['strategy_returns'].dropna()
            if len(strategy_returns) > 1:
                annual_volatility = strategy_returns.std() * np.sqrt(trading_days)
            else:
                annual_volatility = 0

            # Sharpe ratio
            risk_free_rate = 0.0
            if annual_volatility > 0:
                sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            else:
                sharpe_ratio = 0

            # Maximum drawdown
            cumulative_returns = data['cumulative_strategy_returns']
            if len(cumulative_returns) > 0:
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / (1 + running_max)
                max_drawdown = drawdown.min()
            else:
                max_drawdown = 0

            # Win rate and trade statistics
            position_changes = data['position'].diff() != 0
            trade_returns = data.loc[position_changes, 'strategy_returns']

            if len(trade_returns) > 0:
                winning_trades = (trade_returns > 0).sum()
                total_trades = len(trade_returns)
                win_rate = winning_trades / total_trades
            else:
                win_rate = 0
                total_trades = 0

            return {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'annual_volatility': float(annual_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'num_trades': int(total_trades)
            }

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return self._get_zero_metrics()

    def _get_zero_metrics(self):
        """Return zero metrics as fallback"""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'num_trades': 0
        }

    def plot_results(self, backtest_data):
        """
        Plot backtest results

        Args:
            backtest_data (pd.DataFrame): Backtest data from the backtest method

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        try:
            fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

            # Plot price and signals
            axes[0].plot(backtest_data.index, backtest_data['close'], label='Price', linewidth=1)
            axes[0].set_title(f'{self.name} - Price and Signals')

            # Mark buy signals
            buy_signals = backtest_data[backtest_data['signal'] == 1]
            if len(buy_signals) > 0:
                axes[0].scatter(buy_signals.index, buy_signals['close'],
                              marker='^', color='g', s=50, label='Buy', alpha=0.7)

            # Mark sell signals
            sell_signals = backtest_data[backtest_data['signal'] == -1]
            if len(sell_signals) > 0:
                axes[0].scatter(sell_signals.index, sell_signals['close'],
                               marker='v', color='r', s=50, label='Sell', alpha=0.7)

            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylabel('Price')

            # Plot returns comparison
            axes[1].plot(backtest_data.index, backtest_data['cumulative_market_returns'] * 100,
                        label='Market Returns', color='blue', alpha=0.7, linewidth=1)
            axes[1].plot(backtest_data.index, backtest_data['cumulative_strategy_returns'] * 100,
                        label='Strategy Returns', color='green', linewidth=2)
            axes[1].set_title('Cumulative Returns Comparison')
            axes[1].set_ylabel('Returns (%)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Plot drawdown
            if 'cumulative_strategy_returns' in backtest_data.columns:
                cumulative_returns = backtest_data['cumulative_strategy_returns']
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / (1 + running_max) * 100

                axes[2].fill_between(backtest_data.index, 0, drawdown,
                                   color='red', alpha=0.3, label='Drawdown')
                axes[2].set_title('Strategy Drawdown')
                axes[2].set_ylabel('Drawdown (%)')
                axes[2].set_xlabel('Date')

                # Set y-axis limit
                if drawdown.min() < 0:
                    axes[2].set_ylim(drawdown.min() * 1.1, 1)
                else:
                    axes[2].set_ylim(-1, 1)

            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

            plt.tight_layout()

            # Add strategy info as text
            metrics_text = (
                f"Total Return: {backtest_data['cumulative_strategy_returns'].iloc[-1]:.2%}\n"
                f"Max Drawdown: {drawdown.min():.2%}\n"
                f"Signals: {(backtest_data['signal'] != 0).sum()} total"
            )

            fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

            return fig

        except Exception as e:
            self.logger.error(f"Error creating plots: {e}")
            # Return a simple figure with error message
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, f'Error creating plot: {str(e)}',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{self.name} - Plot Error')
            return fig

    def summary(self):
        """
        Get a summary of the strategy

        Returns:
            dict: Strategy summary
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'class': self.__class__.__name__
        }

    def __str__(self):
        """String representation of the strategy"""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={len(self.parameters)})"

    def __repr__(self):
        """Detailed string representation of the strategy"""
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}', parameters={self.parameters})"


# Utility function to validate strategy implementation
def validate_strategy_class(strategy_class):
    """
    Validate that a strategy class properly implements BaseStrategy

    Args:
        strategy_class: The strategy class to validate

    Returns:
        tuple: (is_valid, issues)
    """
    issues = []

    # Check inheritance
    if not issubclass(strategy_class, BaseStrategy):
        issues.append("Strategy class must inherit from BaseStrategy")

    # Check abstract methods are implemented
    try:
        # Try to instantiate (this will fail if abstract methods aren't implemented)
        instance = strategy_class()

        # Check if methods are properly implemented
        import inspect

        # Check _initialize_parameters
        if not hasattr(instance, '_initialize_parameters'):
            issues.append("Missing _initialize_parameters method")
        elif inspect.ismethod(getattr(instance, '_initialize_parameters')):
            # Check if it's not just the abstract method
            try:
                instance._initialize_parameters({})
            except NotImplementedError:
                issues.append("_initialize_parameters method not properly implemented")

        # Check generate_signals
        if not hasattr(instance, 'generate_signals'):
            issues.append("Missing generate_signals method")
        elif inspect.ismethod(getattr(instance, 'generate_signals')):
            # This one is harder to test without data, so just check it exists
            pass

    except TypeError as e:
        if "abstract methods" in str(e):
            # Extract method names from error message
            import re
            methods = re.findall(r"'(\w+)'", str(e))
            issues.append(f"Abstract methods not implemented: {', '.join(methods)}")
        else:
            issues.append(f"Error instantiating strategy: {e}")
    except Exception as e:
        issues.append(f"Error validating strategy: {e}")

    return len(issues) == 0, issues


# Example implementation for testing
class ExampleMovingAverageStrategy(BaseStrategy):
    """
    Example implementation of a moving average crossover strategy
    This shows the correct way to implement BaseStrategy
    """

    def _initialize_parameters(self, params):
        """Initialize strategy parameters"""
        self.fast_window = params.get('fast_window', 10)
        self.slow_window = params.get('slow_window', 50)
        self.name = "Example Moving Average Strategy"
        self.description = "Example moving average crossover strategy"

    def generate_signals(self, data):
        """Generate trading signals based on moving average crossover"""
        df = data.copy()

        # Ensure we have enough data
        if len(df) < max(self.fast_window, self.slow_window):
            df['signal'] = 0
            return df

        # Calculate moving averages
        df['ma_fast'] = df['close'].rolling(window=self.fast_window, min_periods=1).mean()
        df['ma_slow'] = df['close'].rolling(window=self.slow_window, min_periods=1).mean()

        # Generate signals
        df['signal'] = 0  # Default to hold

        # Buy when fast MA crosses above slow MA
        buy_condition = (df['ma_fast'] > df['ma_slow']) & (df['ma_fast'].shift(1) <= df['ma_slow'].shift(1))
        df.loc[buy_condition, 'signal'] = 1

        # Sell when fast MA crosses below slow MA
        sell_condition = (df['ma_fast'] < df['ma_slow']) & (df['ma_fast'].shift(1) >= df['ma_slow'].shift(1))
        df.loc[sell_condition, 'signal'] = -1

        # Fill any NaN values in signal column
        df['signal'] = df['signal'].fillna(0)

        return df