"""
Enhanced BaseStrategy class with improved metadata, validation, and error handling
This fully replaces strategy_format/base_strategy.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import logging
import uuid


class BaseStrategy(ABC):
    """
    Enhanced base class for all trading strategies.
    All LLM-generated strategies should inherit from this class.

    Provides:
    - Standardized interface for all strategies
    - Built-in backtesting functionality
    - Performance metrics calculation
    - Visualization
    - Input validation and error handling
    - Strategy metadata and hyperparameter schema support
    """

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', self.__class__.__name__)
        self.description = kwargs.get('description', '')
        self.parameters = kwargs

        # NEW: strategy metadata fields
        self.origin = kwargs.get('origin', 'manual')   # e.g. "LLM", "human", "genetic"
        self.version = kwargs.get('version', '1.0')
        self.uuid = kwargs.get('uuid', str(uuid.uuid4()))
        self.parameter_schema = kwargs.get('parameter_schema', {})

        # logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        try:
            self._initialize_parameters(kwargs)
            self.logger.info(f"Strategy {self.name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing strategy parameters: {e}")
            raise

    @abstractmethod
    def _initialize_parameters(self, params):
        pass

    @abstractmethod
    def generate_signals(self, data):
        pass

    def get_hyperparameter_space(self):
        """
        Optionally override in a subclass to expose tunable hyperparameters
        for a meta-learner.
        """
        return {}

    def validate_data(self, data):
        if data is None or len(data) == 0:
            raise ValueError("Data is empty or None")

        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data must be a pandas DataFrame, got {type(data)}")

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except:
                    raise ValueError(f"Column {col} cannot be converted to numeric")

        # check OHLC logic
        ohlc_issues = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        if ohlc_issues.any():
            self.logger.warning(f"{ohlc_issues.sum()} invalid OHLC rows found; fixing")
            data.loc[ohlc_issues, 'high'] = data.loc[ohlc_issues, ['open', 'close']].max(axis=1)
            data.loc[ohlc_issues, 'low'] = data.loc[ohlc_issues, ['open', 'close']].min(axis=1)

        # negative volumes
        neg_vol = data['volume'] < 0
        if neg_vol.any():
            self.logger.warning(f"{neg_vol.sum()} negative volumes found; setting to 0")
            data.loc[neg_vol, 'volume'] = 0

        # fill NaNs
        if data[required_columns].isnull().any().any():
            nan_counts = data[required_columns].isnull().sum()
            self.logger.warning(f"Found NaNs: {nan_counts.to_dict()}")
            data[required_columns] = data[required_columns].ffill().bfill()

        return data

    def preprocess_data(self, data):
        data = self.validate_data(data)
        data.columns = [c.lower() for c in data.columns]
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        if data.index.duplicated().any():
            self.logger.warning("Duplicate index entries detected, dropping")
            data = data[~data.index.duplicated(keep='last')]
        return data

    def validate_signals(self, data):
        if 'signal' not in data.columns:
            raise ValueError("Missing 'signal' column")

        valid = [-1, 0, 1]
        bad_signals = ~data['signal'].isin(valid)
        if bad_signals.any():
            self.logger.warning(f"{bad_signals.sum()} invalid signals found; resetting to 0")
            data.loc[bad_signals, 'signal'] = 0

        if data['signal'].isnull().any():
            self.logger.warning("NaN signals found, filling with 0")
            data['signal'] = data['signal'].fillna(0)

        data['signal'] = data['signal'].astype(int)
        return data

    def backtest(self, data):
        try:
            data = self.preprocess_data(data.copy())
            if len(data) < 2:
                raise ValueError("Insufficient data for backtest")

            self.logger.info(f"Generating signals for {len(data)} points")
            data = self.generate_signals(data)
            data = self.validate_signals(data)

            data['position'] = data['signal'].shift(1).fillna(0)
            data['market_returns'] = data['close'].pct_change().fillna(0)
            data['strategy_returns'] = data['position'] * data['market_returns']

            data['cumulative_market_returns'] = (1 + data['market_returns']).cumprod() - 1
            data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod() - 1

            metrics = self._calculate_metrics(data)
            self.logger.info(f"Backtest done. Total return: {metrics['total_return']:.2%}")
            return data, metrics
        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            raise

    def _calculate_metrics(self, data):
        try:
            total = data['cumulative_strategy_returns'].iloc[-1]
            trading_days = 252
            years = len(data) / trading_days
            annual_return = (1 + total) ** (1/years) - 1 if years > 0 else 0

            strategy_returns = data['strategy_returns'].dropna()
            annual_vol = strategy_returns.std() * np.sqrt(trading_days) if len(strategy_returns) > 1 else 0
            sharpe = (annual_return) / annual_vol if annual_vol > 0 else 0

            cumrets = data['cumulative_strategy_returns']
            drawdown = (cumrets - cumrets.expanding().max()) / (1 + cumrets)
            max_dd = drawdown.min()

            trades = data['position'].diff() != 0
            trade_returns = data.loc[trades, 'strategy_returns']
            win_rate = (trade_returns > 0).sum() / len(trade_returns) if len(trade_returns) > 0 else 0

            return {
                'total_return': float(total),
                'annual_return': float(annual_return),
                'annual_volatility': float(annual_vol),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_dd),
                'win_rate': float(win_rate),
                'num_trades': int(trades.sum())
            }
        except Exception as e:
            self.logger.error(f"Metrics error: {e}")
            return self._get_zero_metrics()

    def _get_zero_metrics(self):
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'num_trades': 0
        }

    def plot_results(self, data):
        try:
            fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
            axes[0].plot(data.index, data['close'], label='Price', linewidth=1)
            axes[0].set_title(f"{self.name} - Price and Signals")

            buys = data[data['signal'] == 1]
            sells = data[data['signal'] == -1]
            axes[0].scatter(buys.index, buys['close'], marker='^', color='g', label='Buy', alpha=0.7)
            axes[0].scatter(sells.index, sells['close'], marker='v', color='r', label='Sell', alpha=0.7)
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            axes[0].set_ylabel("Price")

            axes[1].plot(data.index, data['cumulative_market_returns']*100, label="Market", color='blue', alpha=0.7)
            axes[1].plot(data.index, data['cumulative_strategy_returns']*100, label="Strategy", color='green')
            axes[1].set_title("Cumulative Returns")
            axes[1].set_ylabel("Returns (%)")
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            cumrets = data['cumulative_strategy_returns']
            dd = (cumrets - cumrets.expanding().max()) / (1 + cumrets) * 100
            axes[2].fill_between(data.index, 0, dd, color='red', alpha=0.3, label='Drawdown')
            axes[2].set_title("Strategy Drawdown")
            axes[2].set_ylabel("Drawdown (%)")
            axes[2].grid(alpha=0.3)
            axes[2].legend()
            axes[2].set_xlabel("Date")

            plt.tight_layout()
            metrics_text = (
                f"Total Return: {cumrets.iloc[-1]:.2%}\n"
                f"Max Drawdown: {dd.min():.2%}\n"
                f"Signals: {(data['signal'] != 0).sum()} total"
            )
            fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

            return fig
        except Exception as e:
            self.logger.error(f"Plotting error: {e}")
            fig, ax = plt.subplots(figsize=(10,6))
            ax.text(0.5,0.5,f"Plot error: {str(e)}", ha="center", va="center")
            ax.set_title(f"{self.name} - Plot Error")
            return fig

    def summary(self):
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'origin': self.origin,
            'version': self.version,
            'uuid': self.uuid,
            'parameter_schema': self.parameter_schema,
            'class': self.__class__.__name__
        }

    def __str__(self):
        return f"{self.__class__.__name__}(name='{self.name}', parameters={len(self.parameters)})"

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}', parameters={self.parameters})"


# Strategy validation utility
def validate_strategy_class(strategy_class):
    issues = []
    if not issubclass(strategy_class, BaseStrategy):
        issues.append("Strategy class must inherit from BaseStrategy")

    try:
        instance = strategy_class()
        import inspect
        if not hasattr(instance, '_initialize_parameters'):
            issues.append("Missing _initialize_parameters")
        if not hasattr(instance, 'generate_signals'):
            issues.append("Missing generate_signals")
    except TypeError as e:
        if "abstract methods" in str(e):
            import re
            missing = re.findall(r"'(\w+)'", str(e))
            issues.append(f"Abstract methods not implemented: {', '.join(missing)}")
        else:
            issues.append(f"Instantiation error: {e}")
    except Exception as e:
        issues.append(f"Validation error: {e}")
    return len(issues) == 0, issues
