"""
Main script for the Trading Strategy Paper Scraper with LLM integration
"""

import os
import argparse
import logging
import yaml
import sys

# Update imports to use the trading/ directory
from trading.config_manager import ConfigManager
from trading.scraper import ResearchScraper
from trading.analyzer import PaperAnalyzer
from trading.llm_service import LLMService
from trading.strategy_extractor import StrategyExtractor
from trading.validation_service import ValidationService
from trading.strategy_tester import StrategyTester

def setup_logging(log_level, log_file=None):
    """
    Setup logging configuration

    Args:
        log_level (str): Log level (INFO, DEBUG, etc.)
        log_file (str): Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    level = getattr(logging, log_level.upper(), logging.INFO)

    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(level=level, format=log_format,
                           handlers=[
                               logging.FileHandler(log_file),
                               logging.StreamHandler()
                           ])
    else:
        logging.basicConfig(level=level, format=log_format)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Trading Strategy Paper Scraper with LLM')

    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', default='all', choices=['scrape', 'analyze', 'extract', 'test', 'all'],
                       help='Operation mode')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited papers')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of papers to process')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry run mode without making actual API calls')
    parser.add_argument('--model', choices=['haiku', 'sonnet', 'opus'], default='haiku',
                        help='Claude model to use (haiku is cheapest, opus is most expensive)')

    return parser.parse_args()

def check_dependencies():
    """Check if required dependencies are installed"""
    required = ['pandas', 'numpy', 'matplotlib', 'requests', 'bs4', 'nltk', 'tqdm', 'yaml', 'anthropic']

    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install them using pip:")
        print(f"pip install {' '.join(missing)}")
        return False

    # Check for tiktoken (used for token counting)
    try:
        __import__('tiktoken')
    except ImportError:
        print("Warning: tiktoken is not installed. Token counting will use estimations.")
        print("To install: pip install tiktoken")

    return True

def setup_base_strategy():
    """Create the BaseStrategy class file"""
    base_strategy_dir = "strategy_format"
    os.makedirs(base_strategy_dir, exist_ok=True)

    base_strategy_file = os.path.join(base_strategy_dir, "base_strategy.py")

    if not os.path.exists(base_strategy_file):
        print("Creating BaseStrategy class...")
        with open(base_strategy_file, 'w') as f:
            f.write("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    \"\"\"
    Base class for all trading strategies.
    All LLM-generated strategies should inherit from this class.
    \"\"\"
    
    def __init__(self, **kwargs):
        \"\"\"Initialize the strategy with parameters\"\"\"
        # Standard parameters all strategies should have
        self.name = kwargs.get('name', self.__class__.__name__)
        self.description = kwargs.get('description', '')
        self.parameters = kwargs
        
        # Initialize additional parameters
        self._initialize_parameters(kwargs)
    
    @abstractmethod
    def _initialize_parameters(self, params):
        \"\"\"Initialize strategy-specific parameters\"\"\"
        pass
    
    @abstractmethod
    def generate_signals(self, data):
        \"\"\"
        Generate trading signals for the given data
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with added signal column (1=buy, -1=sell, 0=hold)
        \"\"\"
        pass
    
    def preprocess_data(self, data):
        \"\"\"
        Preprocess data before generating signals
        
        Args:
            data (pd.DataFrame): Raw market data
            
        Returns:
            pd.DataFrame: Preprocessed data
        \"\"\"
        # Ensure column names are lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")
            
        return data
    
    def backtest(self, data):
        \"\"\"
        Backtest the strategy on historical data
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            
        Returns:
            tuple: (backtest_data, metrics)
        \"\"\"
        # Preprocess data
        data = self.preprocess_data(data.copy())
        
        # Generate signals
        data = self.generate_signals(data)
        
        # Calculate returns
        data['position'] = data['signal'].shift(1).fillna(0)
        data['market_returns'] = data['close'].pct_change().fillna(0)
        data['strategy_returns'] = data['position'] * data['market_returns']
        
        # Calculate cumulative returns
        data['cumulative_market_returns'] = (1 + data['market_returns']).cumprod() - 1
        data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod() - 1
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(data)
        
        return data, metrics
    
    def _calculate_metrics(self, data):
        \"\"\"Calculate performance metrics\"\"\"
        # Basic metrics
        total_return = data['cumulative_strategy_returns'].iloc[-1]
        
        # Annualized return and volatility
        annual_factor = 252 / len(data) * len(data.index.unique())
        annual_return = (1 + total_return) ** annual_factor - 1
        annual_volatility = data['strategy_returns'].std() * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.0
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = data['cumulative_strategy_returns']
        drawdown = 1 - (1 + cumulative_returns) / (1 + cumulative_returns.cummax())
        max_drawdown = drawdown.max()
        
        # Win rate
        trades = data['position'].diff() != 0
        trade_returns = data.loc[trades, 'strategy_returns']
        win_rate = (trade_returns > 0).sum() / len(trade_returns) if len(trade_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trade_returns)
        }
    
    def plot_results(self, backtest_data):
        \"\"\"
        Plot backtest results
        
        Args:
            backtest_data (pd.DataFrame): Backtest data from the backtest method
        \"\"\"
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot price and signals
        axes[0].plot(backtest_data.index, backtest_data['close'], label='Price')
        axes[0].set_title(f'{self.name} - Price and Signals')
        
        # Mark buy signals
        buy_signals = backtest_data[backtest_data['signal'] == 1].index
        if len(buy_signals) > 0:
            axes[0].scatter(buy_signals, backtest_data.loc[buy_signals, 'close'], 
                          marker='^', color='g', s=100, label='Buy')
        
        # Mark sell signals
        sell_signals = backtest_data[backtest_data['signal'] == -1].index
        if len(sell_signals) > 0:
            axes[0].scatter(sell_signals, backtest_data.loc[sell_signals, 'close'], 
                           marker='v', color='r', s=100, label='Sell')
        
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot returns
        axes[1].plot(backtest_data.index, backtest_data['cumulative_market_returns'], 
                    label='Market Returns', color='blue', alpha=0.7)
        axes[1].plot(backtest_data.index, backtest_data['cumulative_strategy_returns'], 
                    label='Strategy Returns', color='green')
        axes[1].set_title('Cumulative Returns')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot drawdown
        drawdown = 1 - (1 + backtest_data['cumulative_strategy_returns']) / \
                  (1 + backtest_data['cumulative_strategy_returns'].cummax())
        axes[2].fill_between(backtest_data.index, 0, drawdown, color='red', alpha=0.3)
        axes[2].set_title('Drawdown')
        axes[2].set_ylim(0, drawdown.max() * 1.1 if drawdown.max() > 0 else 0.1)
        axes[2].grid(True)
        
        plt.tight_layout()
        return fig
""")

        # Create __init__.py in the directory to make it a package
        with open(os.path.join(base_strategy_dir, "__init__.py"), 'w') as f:
            f.write("# Strategy format package\n")

        print(f"BaseStrategy class created at {base_strategy_file}")
    else:
        print(f"BaseStrategy class already exists at {base_strategy_file}")

    return base_strategy_dir

def run_pipeline(args):
    """
    Run the complete pipeline based on arguments

    Args:
        args: Command line arguments

    Returns:
        dict: Pipeline results
    """
    # Check if required dependencies are installed
    if not check_dependencies():
        return None

    # Ensure BaseStrategy class is available
    setup_base_strategy()

    # Load configuration
    config_manager = ConfigManager(args.config)
    output_dirs = config_manager.get_output_dirs()
    scraping_config = config_manager.get_scraping_config()

    # Apply command line overrides to config
    if args.dry_run:
        if 'llm' not in config_manager.config:
            config_manager.config['llm'] = {}
        config_manager.config['llm']['dry_run'] = True
        print("🔍 DRY RUN MODE ENABLED - No API calls will be made")

    # Set Claude model based on command line argument
    if args.model:
        if 'llm' not in config_manager.config:
            config_manager.config['llm'] = {}
        # FIXED: Updated model identifiers to use correct and valid model names
        model_mapping = {
            'haiku': 'claude-3-5-haiku-20241022',
            'sonnet': 'claude-3-7-sonnet-20250219',
            'opus': 'claude-3-opus-20240229'
        }
        if args.model in model_mapping:
            config_manager.config['llm']['claude_model'] = model_mapping[args.model]
            print(f"Using Claude model: {model_mapping[args.model]}")

    # Setup logging
    log_level = config_manager.get_output_config().get('log_level', 'INFO')
    log_file = os.path.join(output_dirs['base_dir'], 'pipeline.log')
    setup_logging(log_level, log_file)

    # Apply test mode if specified
    if args.test:
        scraping_config['test_mode'] = True
        if args.limit:
            scraping_config['test_paper_limit'] = args.limit

    # Initialize components
    scraper = ResearchScraper(
        query_topics=scraping_config.get('query_topics'),
        max_papers=scraping_config.get('max_papers', 50),
        output_dir=output_dirs['papers_dir'],
        trading_keywords=scraping_config.get('trading_keywords'),
        arxiv_categories=scraping_config.get('arxiv_categories'),
        rate_limit=scraping_config.get('rate_limit', 3)
    )

    analyzer = PaperAnalyzer(
        output_dir=output_dirs['papers_dir']
    )

    # Initialize LLM components
    logging.info("Initializing LLM components")

    # Create LLM config if not present
    llm_config = config_manager.config.get('llm', {})
    if not llm_config:
        llm_config = {
            "primary_provider": "claude",
            "fallback_providers": ["llama3"],
            "cache_dir": os.path.join(output_dirs['base_dir'], "llm_cache"),
            "local_models": {}
        }
        config_manager.config['llm'] = llm_config

    # Initialize LLM service
    llm_service = LLMService(llm_config)

    # Initialize validation service
    validation_service = ValidationService({
        "test_data_path": os.path.join(output_dirs['base_dir'], "test_data.csv")
    })

    # Generate test data if not present
    if not os.path.exists(os.path.join(output_dirs['base_dir'], "test_data.csv")):
        test_data = validation_service.generate_test_data()
        test_data.to_csv(os.path.join(output_dirs['base_dir'], "test_data.csv"))

    # Initialize strategy extractor
    strategies_dir = os.path.join(output_dirs['strategies_dir'], "llm_strategies")
    strategy_extractor = StrategyExtractor(
        llm_service=llm_service,
        validation_service=validation_service,
        config=config_manager.config,
        output_dir=strategies_dir,
        test_mode=args.test
    )

    # Initialize strategy tester
    strategy_tester = StrategyTester(
        strategies_dir=os.path.join(strategies_dir, "implementations"),
        test_data_path=os.path.join(output_dirs['base_dir'], "test_data.csv"),
        results_dir=os.path.join(output_dirs['base_dir'], "test_results")
    )

    # Run the requested operations
    papers = []

    # Scrape papers if requested
    if args.mode in ['scrape', 'all']:
        logging.info("Scraping papers")
        try:
            if scraping_config.get('test_mode', False):
                test_limit = scraping_config.get('test_paper_limit', 5)
                papers = scraper.scrape_papers(test_limit=test_limit)
            else:
                papers = scraper.scrape_papers()

            # Validate that papers were retrieved
            if not papers or len(papers) == 0:
                logging.warning("No papers were retrieved during scraping. Check the scraper configuration.")
        except Exception as e:
            logging.error(f"Error during paper scraping: {e}", exc_info=True)
            papers = []

    # Load papers if not scraping or if needed for subsequent steps
    if (args.mode != 'scrape' or not papers) and args.mode != 'test':
        logging.info("Loading papers")
        try:
            scraper.load_papers(limit=args.limit)
            papers = scraper.papers

            # Validate that papers were loaded
            if not papers or len(papers) == 0:
                logging.warning("No papers were loaded. The paper JSON file may be missing or empty.")

                # If in test mode, create some dummy papers for testing
                if args.test and args.mode in ['analyze', 'extract', 'all']:
                    logging.info("Creating dummy test papers...")
                    papers = [
                        {
                            'id': 'test1',
                            'title': 'Test Momentum Strategy Paper',
                            'abstract': 'This is a test paper about momentum trading strategies.',
                            'authors': ['Test Author'],
                            'published': '2023-01-01',
                            'strategy_type': 'momentum',
                            'asset_class': 'equities'
                        },
                        {
                            'id': 'test2',
                            'title': 'Test Mean Reversion Strategy Paper',
                            'abstract': 'This is a test paper about mean reversion trading strategies.',
                            'authors': ['Test Author'],
                            'published': '2023-02-01',
                            'strategy_type': 'mean reversion',
                            'asset_class': 'forex'
                        }
                    ]
                    scraper.papers = papers
        except Exception as e:
            logging.error(f"Error loading papers: {e}", exc_info=True)

    # Analyze papers if requested
    if args.mode in ['analyze', 'all'] and papers:
        logging.info("Analyzing papers")
        try:
            analyzer.papers = papers
            analyzed_papers = analyzer.analyze_all_papers()

            # Generate statistics and visualizations
            analyzer.generate_summary_statistics()
            analyzer.visualize_statistics()
            analyzer.generate_best_strategy_recommendations()

            # Update papers with analysis
            papers = analyzed_papers
        except Exception as e:
            logging.error(f"Error analyzing papers: {e}", exc_info=True)

    # Extract strategies using LLM
    extracted_strategies = []
    if args.mode in ['extract', 'all'] and papers:
        logging.info("Extracting strategies using LLM")
        try:
            extracted_strategies = strategy_extractor.extract_strategies(papers)

            # Create a catalog of extracted strategies
            strategy_catalog = strategy_extractor.create_strategy_catalog(extracted_strategies)
            logging.info(f"Created strategy catalog with {len(strategy_catalog)} entries")
        except Exception as e:
            logging.error(f"Error extracting strategies: {e}", exc_info=True)

    # Test strategies if requested
    if args.mode in ['test', 'all'] and (extracted_strategies or args.mode == 'test'):
        logging.info("Testing generated strategies")
        try:
            test_results = strategy_tester.test_all_strategies()
            logging.info(f"Tested {len(test_results)} strategies")
        except Exception as e:
            logging.error(f"Error testing strategies: {e}", exc_info=True)

    # Print token usage report if available
    if hasattr(llm_service, 'get_token_usage_report'):
        try:
            usage_report = llm_service.get_token_usage_report()
            logging.info(f"Token usage summary:")
            logging.info(f"  Input tokens: {usage_report['input_tokens']}")
            logging.info(f"  Output tokens: {usage_report['output_tokens']}")
            logging.info(f"  Total tokens: {usage_report['total_tokens']}")
            logging.info(f"  Estimated cost: ${usage_report['estimated_cost']:.4f}")
        except Exception as e:
            logging.error(f"Error generating token usage report: {e}")

    logging.info("Pipeline completed successfully")

    # Return results for potential further processing
    return {
        'papers': papers,
        'scraper': scraper,
        'analyzer': analyzer,
        'llm_service': llm_service,
        'strategy_extractor': strategy_extractor,
        'validation_service': validation_service,
        'strategy_tester': strategy_tester
    }


def main():
    """Main entry point"""
    args = parse_arguments()

    try:
        results = run_pipeline(args)
        if results is None:
            return 1
        return 0
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())