import logging


class StrategyScorer:
    """
    Multi-dimensional scoring of trading strategy papers
    to evaluate implementability.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def score_paper(self, paper):
        """
        Score a paper on multiple dimensions for strategy implementability

        Args:
            paper (dict): Paper dictionary with analysis data

        Returns:
            dict: Scores for different dimensions
        """
        # Initialize scores
        scores = {
            'strategy_completeness': 0.0,
            'implementation_feasibility': 0.0,
            'backtest_quality': 0.0,
            'data_availability': 0.0,
            'complexity': 0.0
        }

        # 1. Strategy Completeness Score
        # Use semantic analysis if available
        if 'semantic_analysis' in paper:
            analysis = paper['semantic_analysis']

            # Core components check
            has_entry = analysis.get('has_entry_rules', False)
            has_exit = analysis.get('has_exit_rules', False)
            has_params = analysis.get('has_parameters', False)
            has_position = analysis.get('has_position_sizing', False)

            # Calculate completeness score
            completeness = sum([
                0.3 if has_entry else 0,
                0.3 if has_exit else 0,
                0.2 if has_params else 0,
                0.2 if has_position else 0
            ])

            # Penalize prediction-only papers
            if analysis.get('is_prediction_only', False):
                completeness *= 0.5

            scores['strategy_completeness'] = completeness
        else:
            # Fallback to structure analysis
            structure = paper.get('structure_analysis', {})

            # Calculate completeness score from structure
            has_trading_rules = structure.get('has_trading_rules', False)
            has_entry_rules = structure.get('has_entry_rules', False)
            has_exit_rules = structure.get('has_exit_rules', False)
            has_thresholds = structure.get('has_thresholds', False)

            completeness = sum([
                0.3 if has_trading_rules else 0,
                0.3 if has_entry_rules or has_exit_rules else 0,
                0.2 if has_thresholds else 0,
                0.2 if len(structure.get('extracted_thresholds', [])) > 0 else 0
            ])

            scores['strategy_completeness'] = completeness

        # 2. Implementation Feasibility Score
        # Assess how feasible the strategy is to implement
        structure = paper.get('structure_analysis', {})

        # Technical indicators give better implementation feasibility
        has_indicators = False
        if 'primary_technique' in paper:
            has_indicators = paper['primary_technique'] not in ['deep_learning', 'reinforcement_learning']

        # Mathematical formulas help with implementation
        has_math_model = structure.get('has_mathematical_model', False)

        # Thresholds are crucial for implementation
        has_thresholds = structure.get('has_thresholds', False)
        num_thresholds = len(structure.get('extracted_thresholds', []))

        # Calculate feasibility score
        feasibility = sum([
            0.3 if has_indicators else 0,
            0.3 if has_math_model else 0,
            0.2 if has_thresholds else 0,
            min(0.2, 0.05 * num_thresholds)  # Up to 0.2 based on number of thresholds
        ])

        scores['implementation_feasibility'] = feasibility

        # 3. Backtest Quality Score
        # Evaluate the quality of backtest results
        has_backtest = structure.get('has_backtest_results', False)

        # Check for performance metrics in semantic analysis
        has_performance_metrics = False
        if 'semantic_analysis' in paper:
            key_elements = paper['semantic_analysis'].get('key_elements', [])
            has_performance_metrics = any('sharpe' in e.lower() or 'return' in e.lower()
                                          for e in key_elements)

        # Calculate backtest quality score
        backtest_quality = sum([
            0.6 if has_backtest else 0,
            0.4 if has_performance_metrics else 0
        ])

        scores['backtest_quality'] = backtest_quality

        # 4. Data Availability Score
        # Estimate how available the required data is
        data_score = 0.0

        # Check asset class if available
        asset_class = paper.get('primary_asset', '').lower()
        if asset_class in ['equity', 'equities', 'stock', 'stocks']:
            data_score += 0.8  # Stock data is widely available
        elif asset_class in ['forex', 'fx', 'currency']:
            data_score += 0.7  # Forex data is quite available
        elif asset_class in ['futures', 'commodities']:
            data_score += 0.6  # Futures data is moderately available
        elif asset_class in ['crypto', 'cryptocurrency']:
            data_score += 0.5  # Crypto data availability varies
        elif asset_class in ['options']:
            data_score += 0.3  # Options data can be hard to get
        else:
            data_score += 0.4  # Default moderate score

        # Adjust for required data frequency if available
        timeframe = paper.get('primary_timeframe', '').lower()
        if timeframe in ['daily', 'day']:
            data_score *= 1.0  # Daily data is easiest
        elif timeframe in ['weekly', 'week']:
            data_score *= 0.9  # Weekly is very easy too
        elif timeframe in ['monthly', 'month']:
            data_score *= 0.95  # Monthly is also very available
        elif timeframe in ['intraday', 'hour', 'minute']:
            data_score *= 0.7  # Intraday can be more challenging
        elif timeframe in ['tick', 'high_frequency']:
            data_score *= 0.4  # Tick data is most difficult

        # Cap at 1.0
        scores['data_availability'] = min(1.0, data_score)

        # 5. Complexity Score (inversely related to complexity)
        # Simpler strategies are preferred for initial implementation
        complexity_score = 0.0

        # Check technique if available
        technique = paper.get('primary_technique', '').lower()
        if technique in ['statistical', 'mean_reversion', 'momentum']:
            complexity_score += 0.8  # Statistical methods are simpler
        elif technique in ['ensemble', 'random_forest']:
            complexity_score += 0.6  # Ensemble methods are moderate
        elif technique in ['deep_learning', 'neural_network']:
            complexity_score += 0.4  # Deep learning is more complex
        elif technique in ['reinforcement_learning']:
            complexity_score += 0.3  # RL is quite complex
        else:
            complexity_score += 0.5  # Default moderate complexity

        # Adjust based on semantic analysis
        if 'semantic_analysis' in paper:
            missing_elements = paper['semantic_analysis'].get('missing_elements', [])
            complexity_score *= (1.0 - 0.1 * min(5, len(missing_elements)))

        scores['complexity'] = complexity_score

        return scores