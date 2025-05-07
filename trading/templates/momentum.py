# Momentum strategy template

TEMPLATE = '''
    def generate_signals(self, data):
        """Generate momentum trading signals"""
        # Calculate momentum over specified lookback period
        data['momentum'] = data['close'].pct_change(self.lookback_period)

        # Generate buy signal when momentum is positive and above threshold
        data['signal'] = 0
        data.loc[data['momentum'] > self.buy_threshold, 'signal'] = 1

        # Generate sell signal when momentum is negative and below threshold
        data.loc[data['momentum'] < self.sell_threshold, 'signal'] = -1

        return data
'''