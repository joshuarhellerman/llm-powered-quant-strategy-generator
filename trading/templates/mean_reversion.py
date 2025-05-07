# Mean reversion strategy template

TEMPLATE = '''
    def generate_signals(self, data):
        """Generate mean reversion trading signals"""
        # Calculate historical mean and standard deviation
        data['rolling_mean'] = data['close'].rolling(window=self.lookback_period).mean()
        data['rolling_std'] = data['close'].rolling(window=self.lookback_period).std()

        # Calculate z-score (deviation from mean in terms of standard deviations)
        data['z_score'] = (data['close'] - data['rolling_mean']) / data['rolling_std']

        # Generate buy signal when price is below mean by threshold
        data['signal'] = 0
        data.loc[data['z_score'] < -self.threshold, 'signal'] = 1

        # Generate sell signal when price is above mean by threshold
        data.loc[data['z_score'] > self.threshold, 'signal'] = -1

        return data
'''