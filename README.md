# Trading Strategy Paper Scraper and Code Generator

## Overview

This system automatically:
1. Scrapes academic papers on trading strategies from arXiv
2. Analyzes the papers to identify strategy types and techniques
3. Generates executable Python code implementing these strategies

The focus is purely on discovering strategies in academic research and converting them to code - no deployment or live trading is included.

## Installation

### Prerequisites
- Python 3.8 or higher
- PyCharm (recommended)

### Setup

1. Clone or download this repository
2. Create a virtual environment:
```bash
python -m venv venv
```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the system with default settings:
```bash
python main.py
```

This will:
- Search arXiv for papers on quantitative trading strategies
- Save paper details to `output/papers/`
- Generate Python code for each strategy in `output/strategies/`

### Command-line Options

```bash
python main.py --help
```

Available options:
- `--query-topics, -q`: Specific topics to search for (e.g., `"momentum trading" "reinforcement learning"`)
- `--max-papers, -m`: Maximum number of papers to scrape (default: 20)
- `--no-scrape, -n`: Skip scraping and use existing papers
- `--output-dir, -o`: Custom output directory

### Examples

Search for specific strategy types:
```bash
python main.py -q "momentum trading" "mean reversion" "reinforcement learning trading"
```

Process more papers:
```bash
python main.py -m 50
```

Use existing papers without re-scraping:
```bash
python main.py -n
```

## Project Structure

```
trading_strategy_extractor/
│
├── requirements.txt
├── README.md
├── main.py
│
├── trading/
│   ├── __init__.py
│   ├── scraper.py          # Research paper scraper
│   ├── generator.py        # Strategy code generator
│   └── templates/          # Strategy templates included in generator.py
│
└── output/
    ├── papers/             # Scraped paper data
    └── strategies/         # Generated Python code
```

## Generated Strategy Code

Each generated strategy file includes:
- Original paper reference and metadata
- Complete strategy implementation with standard methods:
  - `fetch_data()`: Download price data using yfinance
  - `_engineer_features()`: Create technical indicators
  - `generate_signals()`: Core strategy logic from the paper
  - `backtest()`: Simple backtesting functionality
  - `plot_results()`: Visualization of backtest results

## Working with Generated Strategies

You can immediately run any generated strategy file:
```bash
python output/strategies/SomeStrategyName.py
```

Each strategy includes example code that will:
1. Create an instance of the strategy
2. Download S&P 500 data (ticker: SPY)
3. Run a backtest
4. Display performance metrics and charts

## Customization

To add new strategy templates:
1. Edit `trading/generator.py`
2. Add new template methods to the `_load_templates()` function
3. Update the `_determine_template()` method to select your new template type

## Limitations

- The system uses basic pattern matching to identify strategy types
- Generated code is based on templates, not detailed analysis of paper methodology
- Advanced strategies may require manual refinement after generation
- No risk management beyond basic position sizing is implemented

## License

MIT