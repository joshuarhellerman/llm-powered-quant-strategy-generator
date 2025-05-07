"""
Strategy Generator module for creating trading strategy code from paper descriptions
"""

import os
import re
import random
import logging
from tqdm import tqdm


class StrategyGenerator:
    """
    Generates Python code for trading strategies based on paper descriptions
    """

    def __init__(self, papers=None, output_dir="output/strategies", config=None):
        """
        Initialize the StrategyGenerator

        Args:
            papers (list): List of paper dictionaries
            output_dir (str): Directory to save generated strategies
            config (dict): Configuration dictionary
        """
        self.papers = papers or []
        self.output_dir = output_dir
        self.config = config or {}
        self.templates = self._load_templates()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def _load_templates(self):
        """
        Load strategy templates for different types of strategies

        Returns:
            dict: Dictionary of strategy templates
        """
        templates = {
            "momentum": self._load_momentum_template(),
            "mean_reversion": self._load_mean_reversion_template(),
            "reinforcement_learning": self._load_reinforcement_learning_template(),
            "transformer": self._load_transformer_template()
        }

        return templates

    def _load_momentum_template(self):
        """
        Load momentum strategy template

        Returns:
            str: Template code for momentum strategy
        """
        return '''    def generate_signals(self, data):
        """Generate momentum trading signals"""
        # Calculate momentum over specified lookback period
        data['momentum'] = data['close'].pct_change(self.lookback_period)

        # Generate buy signal when momentum is positive and above threshold
        data['signal'] = 0
        data.loc[data['momentum'] > self.buy_threshold, 'signal'] = 1

        # Generate sell signal when momentum is negative and below threshold
        data.loc[data['momentum'] < self.sell_threshold, 'signal'] = -1

        return data'''

    def _load_mean_reversion_template(self):
        """
        Load mean reversion strategy template

        Returns:
            str: Template code for mean reversion strategy
        """
        return '''    def generate_signals(self, data):
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

        return data'''

    def _load_reinforcement_learning_template(self):
        """
        Load reinforcement learning strategy template

        Returns:
            str: Template code for reinforcement learning strategy
        """
        return '''    def train_agent(self, data):
        """Train the reinforcement learning agent"""
        # Feature engineering
        features = self._engineer_features(data)

        # Environment setup
        env = TradingEnvironment(features, initial_balance=self.initial_balance)

        # Agent setup
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size)

        # Training loop
        batch_size = 32
        for e in range(self.episodes):
            state = env.reset()
            done = False

            while not done:
                # Agent selects action
                action = agent.act(state)

                # Take action in environment
                next_state, reward, done, _ = env.step(action)

                # Store experience
                agent.remember(state, action, reward, next_state, done)

                state = next_state

                # Experience replay
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

        return agent

    def generate_signals(self, data):
        """Generate trading signals using the trained agent"""
        # Feature engineering
        features = self._engineer_features(data)

        # Get predictions from agent
        predictions = []
        for i in range(len(features)):
            state = features[i:i+1]
            action = self.agent.act(state, exploit=True)
            predictions.append(action)

        # Convert predictions to signals
        data['signal'] = predictions

        return data'''

    def _load_transformer_template(self):
        """
        Load transformer strategy template

        Returns:
            str: Template code for transformer strategy
        """
        return '''    def build_model(self, input_shape):
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

        return data'''

    def _validate_code_format(self, code):
        """
        Ensure consistent indentation and formatting

        Args:
            code (str): Generated strategy code

        Returns:
            str: Validated and corrected code
        """
        # Split code into lines
        lines = code.split('\n')
        corrected_lines = []

        in_class = False
        class_indent = ''

        for line in lines:
            # Check if this is a class definition
            if line.strip().startswith('class ') and line.strip().endswith(':'):
                in_class = True
                class_indent = line[:len(line) - len(line.lstrip())]
                corrected_lines.append(line)
            # Check if this is an incorrectly indented method
            elif in_class and line.strip().startswith('def ') and not line.startswith(class_indent + '    '):
                # Fix method indentation
                corrected_lines.append(class_indent + '    ' + line.strip())
            else:
                corrected_lines.append(line)

        return '\n'.join(corrected_lines)

    def _determine_template(self, paper):
        """
        Determine which template to use based on paper details

        Args:
            paper (dict): Paper dictionary with metadata

        Returns:
            str: Template name to use
        """
        # Fix: Add check to ensure paper is not None
        if paper is None:
            self.logger.error("Paper is None, using default template")
            return "momentum"  # Default to momentum as a fallback

        # Fix: Add null checks for ml_technique and strategy_type
        ml_technique = paper.get('ml_technique', '')
        if ml_technique is not None:
            ml_technique = ml_technique.lower()
        else:
            ml_technique = ''

        strategy_type = paper.get('strategy_type', '')
        if strategy_type is not None:
            strategy_type = strategy_type.lower()
        else:
            strategy_type = ''

        # Use template weights from config if available
        template_weights = {}
        if self.config and 'generator' in self.config and 'template_weights' in self.config['generator']:
            template_weights = self.config['generator']['template_weights']

        # Default weights
        weights = {
            "momentum": 1.0,
            "mean_reversion": 1.0,
            "reinforcement_learning": 0.8,
            "transformer": 0.7
        }

        # Update with config weights if available
        weights.update(template_weights)

        # Check for explicit strategy indications in the paper
        if 'reinforcement learning' in ml_technique:
            return "reinforcement_learning"
        elif 'transformer' in ml_technique:
            return "transformer"
        elif 'momentum' in strategy_type:
            return "momentum"
        elif 'mean reversion' in strategy_type:
            return "mean_reversion"

        # If no clear indication, use weighted random selection
        template_names = list(weights.keys())
        template_probs = list(weights.values())
        total_weight = sum(template_probs)
        normalized_probs = [w / total_weight for w in template_probs]

        return random.choices(template_names, weights=normalized_probs, k=1)[0]

    def _engineer_template_parameters(self, paper):
        """
        Engineer parameters for the template based on paper details

        Args:
            paper (dict): Paper dictionary with metadata

        Returns:
            dict: Dictionary of parameters for the strategy
        """
        # Fix: Add check to ensure paper is not None
        if paper is None:
            self.logger.error("Paper is None, using default parameters")
            return {
                "lookback_period": 20,
                "buy_threshold": 0.03,
                "sell_threshold": -0.03,
                "threshold": 2.0,
                "initial_balance": 10000,
                "episodes": 100,
                "sequence_length": 20
            }

        # Get parameter ranges from config if available
        param_ranges = {}
        if self.config and 'generator' in self.config and 'parameters' in self.config['generator']:
            param_ranges = self.config['generator']['parameters']

        # Default parameter ranges
        lookback_periods = param_ranges.get('lookback_periods', [5, 10, 20, 50, 100])
        buy_thresholds = param_ranges.get('buy_thresholds', [0.01, 0.02, 0.03, 0.04, 0.05])
        sell_thresholds = param_ranges.get('sell_thresholds', [-0.05, -0.04, -0.03, -0.02, -0.01])
        thresholds = param_ranges.get('thresholds', [1.0, 1.5, 2.0, 2.5])

        # Try to extract parameters from paper if available
        extracted_params = paper.get('parameters', {})
        if extracted_params is None:
            extracted_params = {}

        extracted_lookback = extracted_params.get('lookback_periods', [])
        extracted_thresholds = extracted_params.get('thresholds', [])

        # Use extracted parameters if available, otherwise use random from ranges
        params = {
            "lookback_period": extracted_lookback[0] if extracted_lookback else random.choice(lookback_periods),
            "buy_threshold": max(0.001, extracted_thresholds[0]) if extracted_thresholds else random.choice(buy_thresholds),
            "sell_threshold": min(-0.001, -abs(extracted_thresholds[1] if len(extracted_thresholds) > 1 else extracted_thresholds[0])) if extracted_thresholds else random.choice(sell_thresholds),
            "threshold": abs(extracted_thresholds[0]) if extracted_thresholds else random.choice(thresholds),
            "initial_balance": 10000,
            "episodes": 100,
            "sequence_length": 20
        }

        return params

    def generate_strategy_code(self, paper):
        """
        Generate Python code for a trading strategy based on a paper

        Args:
            paper (dict): Paper dictionary with metadata

        Returns:
            dict: Dictionary with generated strategy information
        """
        # Fix: Add check to ensure paper is not None
        if paper is None:
            self.logger.error("Cannot generate strategy code for None paper")
            return None

        template_name = self._determine_template(paper)
        template = self.templates.get(template_name)

        if not template:
            self.logger.error(f"No template found for {paper.get('title', 'Unknown Paper')}")
            return None

        params = self._engineer_template_parameters(paper)

        # Generate the class name from the paper title
        title = paper.get('title', 'Unknown Paper')
        class_name = ''.join(
            word.capitalize() for word in re.sub(r'[^a-zA-Z0-9\s]', '', title).split()
        ) + 'Strategy'

        # Limit class name length
        if len(class_name) > 50:
            class_name = class_name[:50] + 'Strategy'

        # Generate the strategy code
        strategy_code = f"""
# Trading Strategy based on the paper:
# "{paper.get('title', 'Unknown Paper')}"
# Source: {paper.get('link', 'arXiv')}
# 
# Abstract:
# {paper.get('abstract', 'No abstract available')[:200]}...
#
# Strategy Type: {paper.get('strategy_type', 'N/A')}
# Asset Class: {paper.get('asset_class', 'N/A')}
# ML Technique: {paper.get('ml_technique', 'N/A')}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta


class {class_name}:
    \"\"\"
    Trading strategy based on the paper:
    "{paper.get('title', 'Unknown Paper')}"

    {paper.get('abstract', 'No abstract available')[:200]}...
    \"\"\"

    def __init__(self, lookback_period={params['lookback_period']}, 
                buy_threshold={params['buy_threshold']}, 
                sell_threshold={params['sell_threshold']},
                threshold={params['threshold']},
                initial_balance={params['initial_balance']},
                episodes={params['episodes']},
                sequence_length={params['sequence_length']}):
        self.lookback_period = lookback_period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.threshold = threshold
        self.initial_balance = initial_balance
        self.episodes = episodes
        self.sequence_length = sequence_length
        self.model = None

    def fetch_data(self, ticker, start_date, end_date):
        \"\"\"Fetch historical data for the ticker\"\"\"
        data = yf.download(ticker, start=start_date, end=end_date)
        data.columns = [col.lower() for col in data.columns]
        return data

    def _engineer_features(self, data):
        \"\"\"Engineer features for the model\"\"\"
        features = data.copy()

        # Add price-based features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))

        # Add moving averages
        for window in [5, 10, 20, 50, 200]:
            features[f'ma_{{window}}'] = features['close'].rolling(window=window).mean()
            features[f'ma_{{window}}_ratio'] = features['close'] / features[f'ma_{{window}}']

        # Add volatility
        features['volatility_20'] = features['returns'].rolling(window=20).std()

        # Add technical indicators
        features['rsi_14'] = self._calculate_rsi(features['close'], 14)

        # Drop NaN values
        features = features.dropna()

        return features

    def _calculate_rsi(self, prices, period=14):
        \"\"\"Calculate the Relative Strength Index (RSI)\"\"\"
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _create_sequences(self, data, sequence_length):
        \"\"\"Create sequences for time series models\"\"\"
        sequences = []
        for i in range(len(data) - sequence_length):
            sequence = data.iloc[i:i+sequence_length].values
            sequences.append(sequence)

        return np.array(sequences)

{template}

    def backtest(self, data):
        \"\"\"Backtest the strategy on historical data\"\"\"
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

        metrics = {{
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }}

        return backtest_data, metrics

    def plot_results(self, backtest_data):
        \"\"\"Plot the backtest results\"\"\"
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
    strategy = {class_name}()

    # Fetch data
    data = strategy.fetch_data('SPY', '2020-01-01', '2023-01-01')

    # Backtest strategy
    backtest_data, metrics = strategy.backtest(data)

    # Print metrics
    print("Performance Metrics:")
    print(f"Total Return: {{metrics['total_return']:.2%}}")
    print(f"Sharpe Ratio: {{metrics['sharpe_ratio']:.2f}}")
    print(f"Max Drawdown: {{metrics['max_drawdown']:.2%}}")

    # Plot results
    strategy.plot_results(backtest_data)
"""

        # Validate and correct the code formatting
        validated_code = self._validate_code_format(strategy_code)

        return {
            'paper': paper,
            'strategy_code': validated_code,  # Use validated code
            'class_name': class_name,
            'template_name': template_name
        }

    def generate_strategies(self):
        """
        Generate strategies for all papers

        Returns:
            list: List of generated strategy dictionaries
        """
        if not self.papers:
            self.logger.warning("No papers loaded. Load papers first.")
            return []

        # Fix: Filter out None papers before processing
        valid_papers = [p for p in self.papers if p is not None]
        if len(valid_papers) < len(self.papers):
            self.logger.warning(f"Filtered out {len(self.papers) - len(valid_papers)} None papers")

        strategies = []
        self.logger.info(f"Generating strategies for {len(valid_papers)} papers...")
        for paper in tqdm(valid_papers, desc="Generating strategies"):
            strategy = self.generate_strategy_code(paper)
            if strategy:
                strategies.append(strategy)

                # Save strategy to file
                filename = f"{self.output_dir}/{strategy['class_name']}.py"
                with open(filename, 'w') as f:
                    f.write(strategy['strategy_code'])

        self.logger.info(f"Generated {len(strategies)} strategies in {self.output_dir}/")
        return strategies