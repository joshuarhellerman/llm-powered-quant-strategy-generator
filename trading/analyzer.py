"""
Paper Analysis module for identifying trading strategies in academic papers
"""

import os
import json
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd


class PaperAnalyzer:
    """
    Analyzes research papers to identify specific trading strategies
    """

    def __init__(self, papers=None, output_dir="output/papers"):
        self.papers = papers or []
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Download required NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        # Strategy classification keywords
        self.strategy_keywords = {
            "momentum": [
                "momentum", "trend following", "breakout", "moving average crossover",
                "relative strength", "price acceleration", "momentum factor"
            ],
            "mean_reversion": [
                "mean reversion", "mean-reversion", "reversal", "oscillation",
                "overbought", "oversold", "bollinger bands", "rsi", "relative strength index"
            ],
            "statistical_arbitrage": [
                "statistical arbitrage", "stat arb", "pairs trading", "cointegration",
                "spread trading", "convergence", "divergence", "market neutral"
            ],
            "reinforcement_learning": [
                "reinforcement learning", "rl", "q-learning", "dqn", "deep q network",
                "actor-critic", "policy gradient", "mdp", "markov decision process"
            ],
            "transformer": [
                "transformer", "attention mechanism", "self-attention", "bert",
                "gpt", "nlp", "natural language processing", "sequence modeling"
            ]
        }

        # Technical indicator keywords
        self.technical_indicators = [
            "moving average", "macd", "rsi", "bollinger bands", "atr", "average true range",
            "stochastic oscillator", "obv", "on-balance volume", "ichimoku cloud",
            "fibonacci retracement", "pivot points", "volume profile"
        ]

        # Asset class keywords
        self.asset_classes = {
            "equity": [
                "equity", "equities", "stock", "stocks", "shares", "etf", "etfs",
                "index", "indices"
            ],
            "forex": [
                "forex", "fx", "currency", "currencies", "exchange rate", "eurusd",
                "usdjpy", "gbpusd", "foreign exchange"
            ],
            "crypto": [
                "crypto", "cryptocurrency", "bitcoin", "ethereum", "digital asset",
                "blockchain", "altcoin", "token", "defi"
            ],
            "futures": [
                "futures", "future", "e-mini", "commodity", "commodities",
                "forwards", "softs", "metals", "energies"
            ],
            "options": [
                "options", "option", "calls", "puts", "volatility", "straddle",
                "strangle", "iron condor", "spread", "derivatives"
            ]
        }

        # ML technique keywords
        self.ml_techniques = {
            "deep_learning": [
                "deep learning", "neural network", "cnn", "convolutional", "rnn",
                "recurrent", "lstm", "gru", "feedforward", "backpropagation"
            ],
            "reinforcement_learning": [
                "reinforcement learning", "q-learning", "dqn", "deep q network",
                "policy gradient", "actor-critic", "a2c", "a3c", "ppo", "ddpg"
            ],
            "transformer": [
                "transformer", "attention", "self-attention", "bert", "gpt",
                "sequence-to-sequence", "encoder-decoder", "nlp"
            ],
            "ensemble": [
                "random forest", "decision tree", "xgboost", "gradient boosting",
                "adaboost", "ensemble", "bagging", "boosting"
            ],
            "statistical": [
                "svm", "support vector", "linear regression", "logistic regression",
                "bayesian", "hmm", "hidden markov", "kalman filter", "arima"
            ]
        }

        # Timeframe keywords
        self.timeframes = {
            "high_frequency": [
                "high frequency", "hft", "high-frequency", "microsecond", "millisecond",
                "tick data", "tick-by-tick", "market microstructure", "limit order book"
            ],
            "intraday": [
                "intraday", "day trading", "minute", "hourly", "short-term",
                "scalping", "session", "within-day", "daily"
            ],
            "swing": [
                "swing trading", "swing", "multi-day", "days", "weekly", "short term",
                "intermediate"
            ],
            "position": [
                "position trading", "long term", "long-term", "monthly", "quarterly",
                "yearly", "investment", "buy and hold", "fundamental"
            ]
        }

    def load_papers(self, filename=None):
        """Load papers from JSON file"""
        if filename is None:
            filename = f"{self.output_dir}/trading_papers.json"

        try:
            with open(filename, 'r') as f:
                self.papers = json.load(f)
            print(f"Loaded {len(self.papers)} papers from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found")
            return False

        return True

    def count_keyword_occurrences(self, text, keywords):
        """Count occurrences of keywords in text"""
        text = text.lower()
        counts = {}

        for keyword in keywords:
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
            counts[keyword] = count

        return counts

    def analyze_strategy_types(self, paper):
        """Identify strategy types in a paper"""
        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()

        strategy_scores = {}
        for strategy_type, keywords in self.strategy_keywords.items():
            counts = self.count_keyword_occurrences(text, keywords)
            strategy_scores[strategy_type] = sum(counts.values())

        # Get the strategy with the highest score
        if any(strategy_scores.values()):
            primary_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        else:
            primary_strategy = "unknown"

        # Get secondary strategies
        strategies = [s for s, score in strategy_scores.items() if score > 0]

        return {
            "primary_strategy": primary_strategy,
            "strategies": strategies,
            "strategy_scores": strategy_scores
        }

    def analyze_asset_classes(self, paper):
        """Identify asset classes in a paper"""
        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()

        asset_scores = {}
        for asset_class, keywords in self.asset_classes.items():
            counts = self.count_keyword_occurrences(text, keywords)
            asset_scores[asset_class] = sum(counts.values())

        # Get the asset class with the highest score
        if any(asset_scores.values()):
            primary_asset = max(asset_scores.items(), key=lambda x: x[1])[0]
        else:
            primary_asset = "equity"  # Default to equity if nothing found

        # Get all asset classes mentioned
        assets = [a for a, score in asset_scores.items() if score > 0]

        return {
            "primary_asset": primary_asset,
            "assets": assets,
            "asset_scores": asset_scores
        }

    def analyze_ml_techniques(self, paper):
        """Identify machine learning techniques in a paper"""
        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()

        ml_scores = {}
        for technique, keywords in self.ml_techniques.items():
            counts = self.count_keyword_occurrences(text, keywords)
            ml_scores[technique] = sum(counts.values())

        # Get the ML technique with the highest score
        if any(ml_scores.values()):
            primary_technique = max(ml_scores.items(), key=lambda x: x[1])[0]
        else:
            primary_technique = "unknown"

        # Get all ML techniques mentioned
        techniques = [t for t, score in ml_scores.items() if score > 0]

        return {
            "primary_technique": primary_technique,
            "techniques": techniques,
            "technique_scores": ml_scores
        }

    def analyze_timeframes(self, paper):
        """Identify trading timeframes in a paper"""
        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()

        timeframe_scores = {}
        for timeframe, keywords in self.timeframes.items():
            counts = self.count_keyword_occurrences(text, keywords)
            timeframe_scores[timeframe] = sum(counts.values())

        # Get the timeframe with the highest score
        if any(timeframe_scores.values()):
            primary_timeframe = max(timeframe_scores.items(), key=lambda x: x[1])[0]
        else:
            primary_timeframe = "unknown"

        # Get all timeframes mentioned
        timeframes = [t for t, score in timeframe_scores.items() if score > 0]

        return {
            "primary_timeframe": primary_timeframe,
            "timeframes": timeframes,
            "timeframe_scores": timeframe_scores
        }

    def extract_technical_indicators(self, paper):
        """Extract technical indicators mentioned in a paper"""
        text = (paper.get('title', '') + ' ' + paper.get('abstract', '')).lower()

        indicators = []
        for indicator in self.technical_indicators:
            if re.search(r'\b' + re.escape(indicator) + r'\b', text):
                indicators.append(indicator)

        return indicators

    def extract_parameters(self, paper):
        """Extract potential strategy parameters from a paper"""
        text = paper.get('abstract', '').lower()

        # Look for numbers that might be parameters
        numbers = re.findall(r'\b(\d+)[- ](?:day|period|window|ma|ema|lookback)\b', text)
        numbers.extend(re.findall(r'\b(?:day|period|window|ma|ema|lookback)[- ](\d+)\b', text))

        lookback_periods = [int(n) for n in numbers if n.isdigit()]

        # Look for percentage thresholds
        percentages = re.findall(r'(\d+(?:\.\d+)?)[ ]?%', text)
        thresholds = [float(p) / 100 for p in percentages]

        return {
            "lookback_periods": lookback_periods,
            "thresholds": thresholds
        }

    def analyze_paper(self, paper):
        """Perform full analysis on a single paper"""
        # Get strategy types
        strategy_analysis = self.analyze_strategy_types(paper)

        # Get asset classes
        asset_analysis = self.analyze_asset_classes(paper)

        # Get ML techniques
        ml_analysis = self.analyze_ml_techniques(paper)

        # Get timeframes
        timeframe_analysis = self.analyze_timeframes(paper)

        # Get technical indicators
        indicators = self.extract_technical_indicators(paper)

        # Get parameters
        parameters = self.extract_parameters(paper)

        # Combine all analysis
        analysis = {
            "id": paper.get('id'),
            "title": paper.get('title'),
            "primary_strategy": strategy_analysis["primary_strategy"],
            "strategies": strategy_analysis["strategies"],
            "primary_asset": asset_analysis["primary_asset"],
            "assets": asset_analysis["assets"],
            "primary_technique": ml_analysis["primary_technique"],
            "techniques": ml_analysis["techniques"],
            "primary_timeframe": timeframe_analysis["primary_timeframe"],
            "timeframes": timeframe_analysis["timeframes"],
            "technical_indicators": indicators,
            "parameters": parameters
        }

        return analysis

    def analyze_all_papers(self):
        """Analyze all loaded papers"""
        if not self.papers:
            print("No papers loaded. Load papers first.")
            return []

        print(f"Analyzing {len(self.papers)} papers...")
        analyzed_papers = []

        for paper in self.papers:
            analysis = self.analyze_paper(paper)

            # Update the original paper with analysis
            paper.update(analysis)
            analyzed_papers.append(paper)

            # Save individual paper analysis
            self.save_paper_analysis(paper)

        # Save all papers with analysis
        self.save_all_paper_analysis(analyzed_papers)

        print(f"Analyzed {len(analyzed_papers)} papers.")
        return analyzed_papers

    def save_paper_analysis(self, paper):
        """Save analysis of a single paper"""
        filename = f"{self.output_dir}/{paper['id']}_analysis.json"

        with open(filename, 'w') as f:
            json.dump(paper, f, indent=2)

    def save_all_paper_analysis(self, papers):
        """Save analysis of all papers"""
        filename = f"{self.output_dir}/analyzed_papers.json"

        with open(filename, 'w') as f:
            json.dump(papers, f, indent=2)

        print(f"Saved analyzed papers to {filename}")

    def generate_summary_statistics(self):
        """Generate summary statistics about analyzed papers"""
        if not self.papers:
            print("No papers loaded. Load papers first.")
            return {}

        # Count strategy types
        strategy_counts = {}
        for paper in self.papers:
            strategy = paper.get('primary_strategy', 'unknown')
            # Only add if not None or empty string
            if strategy:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            else:
                # Add to unknown category if None or empty
                strategy_counts["unknown"] = strategy_counts.get("unknown", 0) + 1

        # Count asset classes
        asset_counts = {}
        for paper in self.papers:
            asset = paper.get('primary_asset', 'unknown')
            # Only add if not None or empty string
            if asset:
                asset_counts[asset] = asset_counts.get(asset, 0) + 1
            else:
                # Add to unknown category if None or empty
                asset_counts["unknown"] = asset_counts.get("unknown", 0) + 1

        # Count ML techniques
        ml_counts = {}
        for paper in self.papers:
            technique = paper.get('primary_technique', 'unknown')
            # Only add if not None or empty string
            if technique:
                ml_counts[technique] = ml_counts.get(technique, 0) + 1
            else:
                # Add to unknown category if None or empty
                ml_counts["unknown"] = ml_counts.get("unknown", 0) + 1

        # Count timeframes
        timeframe_counts = {}
        for paper in self.papers:
            timeframe = paper.get('primary_timeframe', 'unknown')
            # Only add if not None or empty string
            if timeframe:
                timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
            else:
                # Add to unknown category if None or empty
                timeframe_counts["unknown"] = timeframe_counts.get("unknown", 0) + 1

        # Calculate average publications per year
        years = {}
        for paper in self.papers:
            if 'published' in paper and paper['published']:
                try:
                    year = paper['published'][:4]  # Extract year from date
                    if year and year != "None":  # Ensure year is not None or 'None' string
                        years[year] = years.get(year, 0) + 1
                except:
                    # Skip if there's an error processing the year
                    pass

        # Generate stats
        stats = {
            "strategy_counts": strategy_counts,
            "asset_counts": asset_counts,
            "ml_counts": ml_counts,
            "timeframe_counts": timeframe_counts,
            "years": years
        }

        # Save stats
        filename = f"{self.output_dir}/paper_statistics.json"
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Saved summary statistics to {filename}")

        return stats

    def visualize_statistics(self, stats=None):
        """Visualize the statistics about analyzed papers"""
        if stats is None:
            stats = self.generate_summary_statistics()

        if not stats:
            print("No statistics to visualize")
            return

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))

        # Plot strategy types
        strategy_counts = stats.get('strategy_counts', {})
        if strategy_counts:
            # Convert to lists for safer plotting
            strategies = list(strategy_counts.keys())
            strategy_values = list(strategy_counts.values())

            # Check if we have valid data to plot
            if strategies and strategy_values and None not in strategies:
                axs[0, 0].bar(strategies, strategy_values)
                axs[0, 0].set_title('Strategy Types')
                axs[0, 0].set_xlabel('Strategy')
                axs[0, 0].set_ylabel('Count')
                axs[0, 0].tick_params(axis='x', rotation=45)
            else:
                axs[0, 0].text(0.5, 0.5, 'No strategy data available',
                              horizontalalignment='center', verticalalignment='center',
                              transform=axs[0, 0].transAxes)

        # Plot asset classes
        asset_counts = stats.get('asset_counts', {})
        if asset_counts:
            # Convert to lists for safer plotting
            assets = list(asset_counts.keys())
            asset_values = list(asset_counts.values())

            # Check if we have valid data to plot
            if assets and asset_values and None not in assets:
                axs[0, 1].bar(assets, asset_values)
                axs[0, 1].set_title('Asset Classes')
                axs[0, 1].set_xlabel('Asset Class')
                axs[0, 1].set_ylabel('Count')
                axs[0, 1].tick_params(axis='x', rotation=45)
            else:
                axs[0, 1].text(0.5, 0.5, 'No asset class data available',
                              horizontalalignment='center', verticalalignment='center',
                              transform=axs[0, 1].transAxes)

        # Plot ML techniques
        ml_counts = stats.get('ml_counts', {})
        if ml_counts:
            # Convert to lists for safer plotting
            ml_techniques = list(ml_counts.keys())
            ml_values = list(ml_counts.values())

            # Check if we have valid data to plot
            if ml_techniques and ml_values and None not in ml_techniques:
                axs[1, 0].bar(ml_techniques, ml_values)
                axs[1, 0].set_title('ML Techniques')
                axs[1, 0].set_xlabel('Technique')
                axs[1, 0].set_ylabel('Count')
                axs[1, 0].tick_params(axis='x', rotation=45)
            else:
                axs[1, 0].text(0.5, 0.5, 'No ML technique data available',
                              horizontalalignment='center', verticalalignment='center',
                              transform=axs[1, 0].transAxes)

        # Plot publications per year
        years = stats.get('years', {})
        if years:
            # Convert to lists for safer plotting
            years_sorted = dict(sorted(years.items()))
            year_keys = list(years_sorted.keys())
            year_values = list(years_sorted.values())

            # Check if we have valid data to plot
            if year_keys and year_values and None not in year_keys:
                axs[1, 1].bar(year_keys, year_values)
                axs[1, 1].set_title('Publications by Year')
                axs[1, 1].set_xlabel('Year')
                axs[1, 1].set_ylabel('Count')
                axs[1, 1].tick_params(axis='x', rotation=45)
            else:
                axs[1, 1].text(0.5, 0.5, 'No publication year data available',
                              horizontalalignment='center', verticalalignment='center',
                              transform=axs[1, 1].transAxes)
        else:
            axs[1, 1].text(0.5, 0.5, 'No publication year data available',
                          horizontalalignment='center', verticalalignment='center',
                          transform=axs[1, 1].transAxes)

        plt.tight_layout()

        # Ensure the output directory exists
        viz_dir = os.path.dirname(f"{self.output_dir}/paper_statistics.png")
        os.makedirs(viz_dir, exist_ok=True)

        # Save the figure
        plt.savefig(f"{self.output_dir}/paper_statistics.png")
        plt.close()

        print(f"Saved visualization to {self.output_dir}/paper_statistics.png")

    def generate_best_strategy_recommendations(self, top_n=5):
        """Generate recommendations for the best strategies to implement"""
        if not self.papers:
            print("No papers loaded. Load papers first.")
            return []

        # Create a scoring system for papers
        paper_scores = []

        for paper in self.papers:
            # Base score
            score = 0

            # Score based on strategy clarity
            if paper.get('primary_strategy', 'unknown') != 'unknown':
                score += 10

            # Score based on asset class clarity
            if paper.get('primary_asset', 'unknown') != 'unknown':
                score += 5

            # Score based on ML technique clarity
            if paper.get('primary_technique', 'unknown') != 'unknown':
                score += 8

            # Score based on specific parameters
            if paper.get('parameters', {}).get('lookback_periods', []):
                score += len(paper.get('parameters', {}).get('lookback_periods', [])) * 3

            if paper.get('parameters', {}).get('thresholds', []):
                score += len(paper.get('parameters', {}).get('thresholds', [])) * 3

            # Score based on technical indicators
            if paper.get('technical_indicators', []):
                score += len(paper.get('technical_indicators', [])) * 2

            paper_scores.append({
                'id': paper.get('id'),
                'title': paper.get('title'),
                'primary_strategy': paper.get('primary_strategy', 'unknown'),
                'primary_asset': paper.get('primary_asset', 'unknown'),
                'primary_technique': paper.get('primary_technique', 'unknown'),
                'score': score
            })

        # Sort papers by score
        paper_scores.sort(key=lambda x: x['score'], reverse=True)

        # Get top N recommendations (or all if less than N)
        recommendations = paper_scores[:min(top_n, len(paper_scores))]

        # Save recommendations
        filename = f"{self.output_dir}/top_strategy_recommendations.json"
        with open(filename, 'w') as f:
            json.dump(recommendations, f, indent=2)

        print(f"Saved top {len(recommendations)} strategy recommendations to {filename}")

        return recommendations