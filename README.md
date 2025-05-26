# Trading Strategy Paper Scraper and LLM Code Generator

## Overview
This project automates the extraction of trading strategies from academic papers and transforms them into executable Python code using Large Language Models (LLMs). It features a complete pipeline from research collection to strategy implementation and testing.

## Key Features

- **Academic Paper Collection**: Scrapes quantitative finance papers from arXiv
- **Multi-Tiered Paper Filtering**: Advanced filtering system to identify papers with complete, implementable trading strategies
- **LLM-Powered Strategy Extraction**: Uses Claude models to identify and implement trading strategies
- **Standardized Implementation**: Generates code that follows a common BaseStrategy interface
- **Backtesting Framework**: Tests strategies against historical market data
- **Visualization Tools**: Generates performance charts and paper statistics

## Installation

### Prerequisites

- Python 3.8 or higher
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - yfinance
  - requests
  - beautifulsoup4
  - nltk
  - tqdm
  - lxml
  - pyyaml
  - anthropic>=0.9.0
  - tiktoken

### Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your Anthropic API key to config.yaml

## Configuration
The system uses config.yaml for configuration with the following main sections:

- **scraping**: Define search parameters like topics, keywords, and categories
- **llm**: Configure Claude model, API keys, and token budget limits
- **output**: Specify directories for saving results
- **analysis**: Set thresholds for strategy identification and recommendation
- **paper_selection**: Configure the multi-tiered filtering system

## Usage

### Basic Usage
Run the complete pipeline:
```bash
python main.py
```

### Command-line Options
```bash
python main.py --help
```

Available options:

- `--config`: Path to configuration file (default: config.yaml)
- `--mode`: Pipeline stage to run (choices: scrape, analyze, extract, test, all)
- `--test`: Run in test mode with limited papers
- `--limit`: Limit number of papers to process
- `--dry-run`: Simulate API calls without making actual requests
- `--model`: Claude model to use (haiku, sonnet, opus)

### Examples
Run only the paper scraping step:
```bash
python main.py --mode scrape
```

Run in test mode with a specific model:
```bash
python main.py --test --model haiku
```

Extract strategies from already scraped papers:
```bash
python main.py --mode extract --limit 5
```

## Project Structure
```
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
│   ├── strategy_tester.py   # Backtesting framework
│   └── paper_selection/     # Advanced filtering system
│       ├── integrator.py    # Pipeline orchestration
│       ├── paper_selector.py # Multi-stage filtering
│       ├── semantic_filter.py # LLM-based strategy validation
│       ├── strategy_scorer.py # Multi-dimensional scoring
│       └── structure_analyzer.py # Document structure analysis
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
```

## Paper Selection System

The project now includes a comprehensive multi-tiered filtering system to identify papers with complete, implementable trading strategies:

### Components

1. **Paper Selection Integrator**: Orchestrates the entire filtering pipeline with configurable thresholds and test modes
2. **Paper Selector**: Multi-stage filtering with basic keyword filtering, structure analysis, semantic evaluation, and final scoring
3. **Semantic Filter**: Uses LLM evaluation to distinguish complete trading strategies from prediction-only research papers
4. **Strategy Scorer**: Multi-dimensional scoring across strategy completeness, implementation feasibility, backtest quality, data availability, and complexity
5. **Structure Analyzer**: Identifies key sections (trading rules, backtest results, risk management) and extracts specific trading elements

### Filtering Stages

1. **Basic Filter**: Keyword-based initial screening
2. **Structure Analysis**: Document structure evaluation for key trading strategy sections
3. **Semantic Filter**: LLM-powered evaluation of strategy completeness
4. **Final Scoring**: Multi-dimensional ranking for implementability

## LLM Strategy Generation
The system uses a two-stage approach to generate strategies:

1. **Strategy Extraction**: Analyzes papers to extract key components:
   - Strategy name and core mechanism
   - Technical indicators and parameters
   - Mathematical formulas
   - Asset classes and market conditions
   - Risk management rules

2. **Code Generation**: Creates Python implementation that:
   - Inherits from BaseStrategy
   - Implements required interface methods
   - Calculates indicators and generates signals
   - Handles preprocessing and validation

## Known Limitations

- **Research Interpretation**: The system sometimes misinterprets research papers about prediction techniques as trading strategies
- **Strategy Directionality**: Generated strategies may lack proper directional components (buy/sell logic)
- **Paper Selection**: Academic papers often focus on forecasting methods rather than complete trading strategies
- **Model Constraints**: Strategy quality depends heavily on the capabilities of the selected Claude model

## Next Steps

- Improve paper selection to better identify true trading strategy research
- Enhance validation to verify strategies contain proper directional components
- Refine LLM prompts to prevent "strategy invention" from prediction papers
- Add comprehensive strategy comparison and performance visualization tools

