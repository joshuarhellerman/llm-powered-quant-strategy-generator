import numpy as np
import pandas as pd
import tensorflow as tf
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from strategy_format.base_strategy import BaseStrategy

class HybridVolatilityStrategy(BaseStrategy):
    """
    Hybrid S&P 500 Volatility Forecasting Strategy
    
    A multi-model volatility prediction strategy using ensemble machine learning 
    and statistical techniques to forecast S&P 500 index volatility.
    
    Key Components:
    1. VIX Index Integration
    2. GARCH Volatility Modeling
    3. LSTM Neural Network Volatility Prediction
    """
    
    def _initialize_parameters(self, params):
        """
        Initialize strategy-specific parameters with robust defaults
        
        Args:
            params (dict): Configuration parameters for the strategy
        """
        # Volatility modeling parameters
        self.vix_weight = params.get('vix_weight', 0.3)
        self.garch_weight = params.get('garch_weight', 0.3)
        self.lstm_weight = params.get('lstm_weight', 0.4)
        
        # GARCH model parameters
        self.garch_p = params.get('garch_p', 1)
        self.garch_q = params.get('garch_q', 1)
        
        # LSTM model parameters
        self.lstm_units = params.get('lstm_units', 50)
        self.lookback_period = params.get('lookback_period', 20)
        self.validation_split = params.get('validation_split', 0.2)
        
        # Risk management thresholds
        self.volatility_threshold = params.get('volatility_threshold', 0.15)
        self.confidence_interval = params.get('confidence_interval', 0.95)
    
    def _preprocess_data(self, data):
        """
        Preprocess input data for volatility modeling
        
        Args:
            data (pd.DataFrame): Input financial time series data
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Calculate log returns
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Handle potential NaN values
        data.dropna(inplace=True)
        
        return data
    
    def _garch_volatility_forecast(self, returns):
        """
        GARCH volatility forecast
        
        Args:
            returns (np.array): Log returns
        
        Returns:
            float: Forecasted volatility
        """
        try:
            # Fit GARCH model
            garch_model = arch_model(returns, 
                                     p=self.garch_p, 
                                     q=self.garch_q).fit()
            
            # Forecast volatility
            forecast = garch_model.forecast(horizon=1)
            return forecast.variance.values[-1][0]
        except Exception as e:
            print(f"GARCH Forecast Error: {e}")
            return np.nan
    
    def _lstm_volatility_model(self, data):
        """
        Create and train LSTM volatility prediction model
        
        Args:
            data (pd.DataFrame): Input time series data
        
        Returns:
            tf.keras.Model: Trained LSTM model
        """
        # Prepare LSTM input data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data['log_returns'].values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.lookback_period):
            X.append(scaled_data[i:i+self.lookback_period])
            y.append(scaled_data[i+self.lookback_period])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Build LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(self.lstm_units, input_shape=(X_train.shape[1], 1)),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, verbose=0)
        
        return model, scaler
    
    def generate_signals(self, data):
        """
        Generate trading signals based on hybrid volatility forecast
        
        Args:
            data (pd.DataFrame): Input financial time series data
        
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        # Preprocess data
        df = self._preprocess_data(data)
        
        # Compute individual volatility forecasts
        garch_vol = self._garch_volatility_forecast(df['log_returns'])
        
        # LSTM volatility forecast
        lstm_model, scaler = self._lstm_volatility_model(df)
        lstm_input = scaler.transform(df['log_returns'].values.reshape(-1, 1))
        lstm_vol = lstm_model.predict(lstm_input[-self.lookback_period:].reshape(1, self.lookback_period, 1))[0][0]
        
        # VIX integration (assuming VIX column exists)
        vix_vol = df['vix'].iloc[-1] / 100.0  # Normalize VIX
        
        # Hybrid volatility forecast
        hybrid_vol = (
            self.garch_weight * garch_vol + 
            self.lstm_weight * lstm_vol + 
            self.vix_weight * vix_vol
        )
        
        # Signal generation logic
        df['volatility'] = hybrid_vol
        df['signal'] = 0  # Default to hold
        
        # Generate signals based on volatility
        df.loc[hybrid_vol > self.volatility_threshold, 'signal'] = -1  # High volatility = sell
        df.loc[hybrid_vol < self.volatility_threshold * 0.5, 'signal'] = 1  # Low volatility = buy
        
        return df