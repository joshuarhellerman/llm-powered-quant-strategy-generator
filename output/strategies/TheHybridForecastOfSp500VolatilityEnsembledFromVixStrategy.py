
# Trading Strategy based on the paper:
# "The Hybrid Forecast of S&P 500 Volatility ensembled from VIX, GARCH and
  LSTM models"
# Source: http://arxiv.org/abs/2407.16780v1
# 
# Abstract:
# Predicting the S&P 500 index volatility is crucial for investors and
financial analysts as it helps assess market risk and make informed investment
decisions. Volatility represents the level of uncert...
#
# Strategy Type: None
# Asset Class: None
# ML Technique: deep learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta


class TheHybridForecastOfSp500VolatilityEnsembledFromVixStrategy:
    """
    Trading strategy based on the paper:
    "The Hybrid Forecast of S&P 500 Volatility ensembled from VIX, GARCH and
  LSTM models"

    Predicting the S&P 500 index volatility is crucial for investors and
financial analysts as it helps assess market risk and make informed investment
decisions. Volatility represents the level of uncert...
    """

    def __init__(self, lookback_period=50, 
                buy_threshold=0.02, 
                sell_threshold=-0.04,
                threshold=1.0,
                initial_balance=10000,
                episodes=100,
                sequence_length=20):
        self.lookback_period = lookback_period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.threshold = threshold
        self.initial_balance = initial_balance
        self.episodes = episodes
        self.sequence_length = sequence_length
        self.model = None

    def fetch_data(self, ticker, start_date, end_date):
        """Fetch historical data for the ticker"""
        data = yf.download(ticker, start=start_date, end=end_date)
        data.columns = [col.lower() for col in data.columns]
        return data

    def _engineer_features(self, data):
        """Engineer features for the model"""
        features = data.copy()

        # Add price-based features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))

        # Add moving averages
        for window in [5, 10, 20, 50, 200]:
            features[f'ma_{window}'] = features['close'].rolling(window=window).mean()
            features[f'ma_{window}_ratio'] = features['close'] / features[f'ma_{window}']

        # Add volatility
        features['volatility_20'] = features['returns'].rolling(window=20).std()

        # Add technical indicators
        features['rsi_14'] = self._calculate_rsi(features['close'], 14)

        # Drop NaN values
        features = features.dropna()

        return features

    def _calculate_rsi(self, prices, period=14):
        """Calculate the Relative Strength Index (RSI)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _create_sequences(self, data, sequence_length):
        """Create sequences for time series models"""
        sequences = []
        for i in range(len(data) - sequence_length):
            sequence = data.iloc[i:i+sequence_length].values
            sequences.append(sequence)

        return np.array(sequences)

    def build_model(self, input_shape):
        """Build a transformer-based model for time series prediction"""
        # Define input layer
        inputs = Input(shape=input_shape)

        # Add positional encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        pos_encoding = self.positional_encoding(positions, input_shape[1])
        x = inputs + pos_encoding

        # Multi-head attention block
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=64
        )(x, x)
        attention_output = Dropout(0.1)(attention_output)
        out1 = LayerNormalization()(inputs + attention_output)

        # Feed forward network
        ffn_output = Dense(128, activation="relu")(out1)
        ffn_output = Dense(input_shape[1])(ffn_output)
        ffn_output = Dropout(0.1)(ffn_output)
        out2 = LayerNormalization()(out1 + ffn_output)

        # Output layer
        output = GlobalAveragePooling1D()(out2)
        output = Dense(64, activation="relu")(output)
        output = Dense(1, activation="tanh")(output)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer="adam", loss="mse")

        return model

    def generate_signals(self, data):
        """Generate trading signals using the transformer model"""
        # Feature engineering
        features = self._engineer_features(data)

        # Generate predictions
        X = self._create_sequences(features, self.sequence_length)
        predictions = self.model.predict(X)

        # Convert predictions to signals
        data = data.iloc[self.sequence_length:]
        data["prediction"] = predictions.flatten()

        # Generate buy signals
        data["signal"] = 0
        data.loc[data["prediction"] > self.buy_threshold, "signal"] = 1

        # Generate sell signals
        data.loc[data["prediction"] < self.sell_threshold, "signal"] = -1

        return data

    def backtest(self, data):
        """Backtest the strategy on historical data"""
        # Make a copy of the data
        backtest_data = data.copy()

        # Generate trading signals
        backtest_data = self.generate_signals(backtest_data)

        # Calculate strategy returns
        backtest_data['position'] = backtest_data['signal'].shift(1)
        backtest_data['returns'] = backtest_data['close'].pct_change()
        backtest_data['strategy_returns'] = backtest_data['position'] * backtest_data['returns']

        # Calculate cumulative returns
        backtest_data['cumulative_returns'] = (1 + backtest_data['returns']).cumprod()
        backtest_data['cumulative_strategy_returns'] = (1 + backtest_data['strategy_returns']).cumprod()

        # Calculate performance metrics
        total_return = backtest_data['cumulative_strategy_returns'].iloc[-1] - 1
        sharpe_ratio = backtest_data['strategy_returns'].mean() / backtest_data['strategy_returns'].std() * (252 ** 0.5)
        max_drawdown = (backtest_data['cumulative_strategy_returns'] / backtest_data['cumulative_strategy_returns'].cummax() - 1).min()

        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

        return backtest_data, metrics

    def plot_results(self, backtest_data):
        """Plot the backtest results"""
        plt.figure(figsize=(12, 8))

        # Plot price and signals
        plt.subplot(3, 1, 1)
        plt.plot(backtest_data['close'])
        plt.title('Price and Signals')

        # Mark buy signals
        buy_signals = backtest_data[backtest_data['signal'] == 1].index
        plt.plot(buy_signals, backtest_data.loc[buy_signals, 'close'], '^', markersize=10, color='g')

        # Mark sell signals
        sell_signals = backtest_data[backtest_data['signal'] == -1].index
        plt.plot(sell_signals, backtest_data.loc[sell_signals, 'close'], 'v', markersize=10, color='r')

        # Plot strategy returns
        plt.subplot(3, 1, 2)
        plt.plot(backtest_data['cumulative_returns'], label='Buy and Hold')
        plt.plot(backtest_data['cumulative_strategy_returns'], label='Strategy')
        plt.title('Cumulative Returns')
        plt.legend()

        # Plot drawdown
        plt.subplot(3, 1, 3)
        plt.plot(backtest_data['cumulative_strategy_returns'] / backtest_data['cumulative_strategy_returns'].cummax() - 1)
        plt.title('Drawdown')

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create strategy
    strategy = TheHybridForecastOfSp500VolatilityEnsembledFromVixStrategy()

    # Fetch data
    data = strategy.fetch_data('SPY', '2020-01-01', '2023-01-01')

    # Backtest strategy
    backtest_data, metrics = strategy.backtest(data)

    # Print metrics
    print("Performance Metrics:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

    # Plot results
    strategy.plot_results(backtest_data)
