# Strategy Format Specification

All generated trading strategies must adhere to this format specification for compatibility with the testing framework.

## Class Structure

Each strategy must:

1. Inherit from `BaseStrategy`
2. Implement required abstract methods
3. Follow the standard method signatures
4. Include proper documentation

## Required Methods

### `_initialize_parameters(self, params)`

Initialize strategy-specific parameters from the provided params dictionary.

### `generate_signals(self, data)`

Generate trading signals based on the strategy logic:
- Input: DataFrame with OHLCV data
- Output: Same DataFrame with added 'signal' column (1=buy, -1=sell, 0=hold)

## Optional Methods

Strategies may override these methods for custom behavior:

### `preprocess_data(self, data)`

Custom data preprocessing specific to the strategy.

### `calculate_metrics(self, data)`

Custom performance metrics calculation.

## Example Strategy Structure

```python
class MyStrategy(BaseStrategy):
    """
    Strategy description goes here.
    
    Based on: [Paper Title]
    Authors: [Paper Authors]
    """
    
    def _initialize_parameters(self, params):
        """Initialize strategy parameters"""
        self.lookback_period = params.get('lookback_period', 20)
        self.buy_threshold = params.get('buy_threshold', 0.05)
        self.sell_threshold = params.get('sell_threshold', -0.05)
    
    def generate_signals(self, data):
        """Generate trading signals"""
        # Strategy logic here
        data['signal'] = 0  # Default to hold
        
        # Buy/sell logic
        # ...
        
        return data