Trading Strategy Paper Scraper and LLM Code Generator
Overview
This project automates the extraction of trading strategies from academic papers and transforms them into executable Python code using Large Language Models (LLMs). It features a complete pipeline from research collection to strategy implementation and testing.
Key Features

Academic Paper Collection: Scrapes quantitative finance papers from arXiv
LLM-Powered Strategy Extraction: Uses Claude models to identify and implement trading strategies
Standardized Implementation: Generates code that follows a common BaseStrategy interface
Backtesting Framework: Tests strategies against historical market data
Visualization Tools: Generates performance charts and paper statistics

Installation
Prerequisites

Python 3.8 or higher
Required packages:

pandas
numpy
matplotlib
yfinance
requests
beautifulsoup4
nltk
tqdm
lxml
pyyaml
anthropic>=0.9.0
tiktoken
Setup

Clone the repository
Create and activate a virtual environment:
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
bashpip install -r requirements.txt

Add your Anthropic API key to config.yaml

Configuration
The system uses config.yaml for configuration with the following main sections:

scraping: Define search parameters like topics, keywords, and categories
llm: Configure Claude model, API keys, and token budget limits
output: Specify directories for saving results
analysis: Set thresholds for strategy identification and recommendation

Usage
Basic Usage
Run the complete pipeline:
bashpython main.py
Command-line Options
bashpython main.py --help
Available options:

--config: Path to configuration file (default: config.yaml)
--mode: Pipeline stage to run (choices: scrape, analyze, extract, test, all)
--test: Run in test mode with limited papers
--limit: Limit number of papers to process
--dry-run: Simulate API calls without making actual requests
--model: Claude model to use (haiku, sonnet, opus)

Examples
Run only the paper scraping step:
bashpython main.py --mode scrape
Run in test mode with a specific model:
bashpython main.py --test --model haiku
Extract strategies from already scraped papers:
bashpython main.py --mode extract --limit 5
Project Structure
trading_strategy_extractor/
│
├── main.py                  # Main entry point
├── config.yaml              # Configuration file
├── requirements.txt         # Dependencies
│
├── trading/                 # Core modules
│   ├── config_manager.py    # Configuration handling
│   ├── scraper.py           # arXiv paper collection
│   ├── analyzer.py          # Paper analysis and statistics
│   ├── llm_service.py       # Claude API integration with token tracking
│   ├── strategy_extractor.py # Strategy extraction and code generation
│   ├── validation_service.py # Code validation and verification
│   └── strategy_tester.py   # Backtesting framework
│
├── strategy_format/         # Strategy template structure
│   ├── base_strategy.py     # BaseStrategy class definition
│   └── __init__.py
│
└── output/                  # Generated files
    ├── papers/              # Scraped papers and metadata
    │   └── papers_YYYYMMDD/ # Daily paper archives
    ├── strategies/          # Generated strategy implementations
    │   └── llm_strategies/  # LLM-generated strategies
    │       ├── overviews/   # Strategy overview JSONs
    │       ├── implementations/ # Python implementation files
    │       └── validation_reports/ # Validation results
    ├── test_results/        # Backtest performance metrics
    └── visualizations/      # Charts and analysis plots
LLM Strategy Generation
The system uses a two-stage approach to generate strategies:

Strategy Extraction: Analyzes papers to extract key components:

Strategy name and core mechanism
Technical indicators and parameters
Mathematical formulas
Asset classes and market conditions
Risk management rules


Code Generation: Creates Python implementation that:

Inherits from BaseStrategy
Implements required interface methods
Calculates indicators and generates signals
Handles preprocessing and validation



Known Limitations

Research Interpretation: The system sometimes misinterprets research papers about prediction techniques as trading strategies
Strategy Directionality: Generated strategies may lack proper directional components (buy/sell logic)
Paper Selection: Academic papers often focus on forecasting methods rather than complete trading strategies
Model Constraints: Strategy quality depends heavily on the capabilities of the selected Claude model

Next Steps

Improve paper selection to better identify true trading strategy research
Enhance validation to verify strategies contain proper directional components
Refine LLM prompts to prevent "strategy invention" from prediction papers
Add comprehensive strategy comparison and performance visualization tools

License
MIT
