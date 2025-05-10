# strategy_format/base_strategy.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    All LLM-generated strategies should inherit from this class.
    """

    def __init__(self, **kwargs):
        """Initialize the strategy with parameters"""
        # Standard parameters all strategies should have
        self.name = kwargs.get('name', self.__class__.__name__)
        self.description = kwargs.get('description', '')
        self.parameters = kwargs

        # Initialize additional parameters
        self._initialize_parameters(kwargs)

    @abstractmethod
    def _initialize_parameters(self, params):
        """Initialize strategy-specific parameters"""
        pass

    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals for the given data

        Args:
            data (pd.DataFrame): Market data with OHLCV columns

        Returns:
            pd.DataFrame: Data with added signal column (1=buy, -1=sell, 0=hold)
        """
        pass

    def preprocess_data(self, data):
        """
        Preprocess data before generating signals

        Args:
            data (pd.DataFrame): Raw market data

        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Ensure column names are lowercase
        data.columns = [col.lower() for col in data.columns]

        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")

        return data

    def backtest(self, data):
        """
        Backtest the strategy on historical data

        Args:
            data (pd.DataFrame): Market data with OHLCV columns

        Returns:
            tuple: (backtest_data, metrics)
        """
        # Preprocess data
        data = self.preprocess_data(data.copy())

        # Generate signals
        data = self.generate_signals(data)

        # Calculate returns
        data['position'] = data['signal'].shift(1).fillna(0)
        data['market_returns'] = data['close'].pct_change().fillna(0)
        data['strategy_returns'] = data['position'] * data['market_returns']

        # Calculate cumulative returns
        data['cumulative_market_returns'] = (1 + data['market_returns']).cumprod() - 1
        data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod() - 1

        # Calculate performance metrics
        metrics = self._calculate_metrics(data)

        return data, metrics

    def _calculate_metrics(self, data):
        """Calculate performance metrics"""
        # Basic metrics
        total_return = data['cumulative_strategy_returns'].iloc[-1]

        # Annualized return and volatility
        annual_factor = 252 / len(data) * len(data.index.unique())
        annual_return = (1 + total_return) ** annual_factor - 1
        annual_volatility = data['strategy_returns'].std() * np.sqrt(252)

        # Sharpe ratio
        risk_free_rate = 0.0
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        # Maximum drawdown
        cumulative_returns = data['cumulative_strategy_returns']
        drawdown = 1 - (1 + cumulative_returns) / (1 + cumulative_returns.cummax())
        max_drawdown = drawdown.max()

        # Win rate
        trades = data['position'].diff() != 0
        trade_returns = data.loc[trades, 'strategy_returns']
        win_rate = (trade_returns > 0).sum() / len(trade_returns) if len(trade_returns) > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trade_returns)
        }

    def plot_results(self, backtest_data):
        """
        Plot backtest results

        Args:
            backtest_data (pd.DataFrame): Backtest data from the backtest method
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # Plot price and signals
        axes[0].plot(backtest_data.index, backtest_data['close'], label='Price')
        axes[0].set_title(f'{self.name} - Price and Signals')

        # Mark buy signals
        buy_signals = backtest_data[backtest_data['signal'] == 1].index
        axes[0].scatter(buy_signals, backtest_data.loc[buy_signals, 'close'],
                        marker='^', color='g', s=100, label='Buy')

        # Mark sell signals
        sell_signals = backtest_data[backtest_data['signal'] == -1].index
        axes[0].scatter(sell_signals, backtest_data.loc[sell_signals, 'close'],
                        marker='v', color='r', s=100, label='Sell')

        axes[0].legend()
        axes[0].grid(True)

        # Plot returns
        axes[1].plot(backtest_data.index, backtest_data['cumulative_market_returns'],
                     label='Market Returns', color='blue', alpha=0.7)
        axes[1].plot(backtest_data.index, backtest_data['cumulative_strategy_returns'],
                     label='Strategy Returns', color='green')
        axes[1].set_title('Cumulative Returns')
        axes[1].legend()
        axes[1].grid(True)

        # Plot drawdown
        drawdown = 1 - (1 + backtest_data['cumulative_strategy_returns']) / \
                   (1 + backtest_data['cumulative_strategy_returns'].cummax())
        axes[2].fill_between(backtest_data.index, 0, drawdown, color='red', alpha=0.3)
        axes[2].set_title('Drawdown')
        axes[2].set_ylim(0, drawdown.max() * 1.1)
        axes[2].grid(True)

        plt.tight_layout()
        return fig