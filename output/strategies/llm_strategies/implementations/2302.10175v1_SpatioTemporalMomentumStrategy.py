import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'strategy_format'))
from base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class SpatioTemporalMomentumStrategy(BaseStrategy):
    """
    Spatio-Temporal Momentum Strategy
    
    A neural network-based approach that generates trading signals by 
    incorporating both time-series and cross-sectional momentum features 
    across multiple assets.
    
    Based on research paper: Spatio-Temporal Momentum: Jointly Learning 
    Time-Series and Cross-Sectional Strategies
    """

    def _initialize_parameters(self, params):
        """
        Initialize strategy parameters with sensible defaults
        
        :param params: Dictionary of strategy parameters
        """
        # Time-series momentum parameters
        self.ts_lookback = params.get('ts_lookback', 20)  # Default 20-day lookback
        
        # Cross-sectional momentum parameters
        self.cs_lookback = params.get('cs_lookback', 30)  # Default 30-day lookback
        
        # Neural network-inspired signal generation parameters
        self.momentum_weight = params.get('momentum_weight', 0.5)
        
        # Strategy metadata
        self.name = "Spatio-Temporal Momentum Strategy"
        self.description = "Neural network-based momentum strategy across assets"

    def _calculate_time_series_momentum(self, close_prices):
        """
        Calculate time-series momentum for individual asset
        
        :param close_prices: Closing prices series
        :return: Time-series momentum score
        """
        if len(close_prices) < self.ts_lookback:
            return 0
        
        # Calculate rolling returns
        returns = close_prices.pct_change(periods=self.ts_lookback)
        
        # Return the most recent momentum score
        return returns.iloc[-1]

    def _calculate_cross_sectional_momentum(self, close_prices_df):
        """
        Calculate cross-sectional momentum across assets
        
        :param close_prices_df: DataFrame of closing prices for multiple assets
        :return: Cross-sectional momentum scores
        """
        if len(close_prices_df) < self.cs_lookback:
            return pd.Series(0, index=close_prices_df.columns)
        
        # Calculate rolling returns for each asset
        returns = close_prices_df.pct_change(periods=self.cs_lookback)
        
        # Use the most recent period's returns
        recent_returns = returns.iloc[-1]
        
        # Rank returns (higher rank = stronger momentum)
        return recent_returns.rank(ascending=False)

    def _generate_neural_signal(self, ts_momentum, cs_momentum):
        """
        Simplified neural network-inspired signal generation
        
        :param ts_momentum: Time-series momentum score
        :param cs_momentum: Cross-sectional momentum score
        :return: Trading signal (-1, 0, 1)
        """
        # Simple weighted combination mimicking a single layer neural network
        combined_score = (self.momentum_weight * ts_momentum + 
                          (1 - self.momentum_weight) * cs_momentum)
        
        # Signal generation threshold
        if combined_score > 0.5:
            return 1  # Buy
        elif combined_score < -0.5:
            return -1  # Sell
        else:
            return 0  # Hold

    def generate_signals(self, data):
        """
        Generate trading signals based on spatio-temporal momentum
        
        :param data: DataFrame with OHLCV data
        :return: DataFrame with trading signals
        """
        df = data.copy()
        
        # Ensure sufficient data
        if len(df) < max(self.ts_lookback, self.cs_lookback):
            df['signal'] = 0
            return df
        
        # Calculate time-series momentum
        ts_momentum = self._calculate_time_series_momentum(df['close'])
        
        # For cross-sectional, you would typically have multiple assets
        # This is a simplified single-asset version
        cs_momentum = self._calculate_cross_sectional_momentum(
            pd.DataFrame(df['close'])
        )
        
        # Generate signal
        signal = self._generate_neural_signal(ts_momentum, cs_momentum)
        
        # Assign signal
        df['signal'] = signal
        
        # Fill any potential NaN values
        df['signal'] = df['signal'].fillna(0)
        
        return df