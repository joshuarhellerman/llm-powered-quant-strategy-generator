# System Analysis: quant_strategy_scraper

## Overview

- **Project Directory**: .
- **Files**: 55
- **Directories**: 18

## Important Files

### Requirements

- `requirements.txt`

### Readme

- `README.md`

### Entry_point

- `main.py`

## Project Structure

```
quant_strategy_scraper/
├── .gitignore
├── Configurations Guide
├── README.md
├── analyze_system.py
├── api_test.py
├── config.yaml
├── debug.py
├── diagnose.py
├── main.py
├── pipeline_stages.py
├── requirements.txt
├── system_analysis.md
├── cache/
├── output/
│   ├── pipeline.log
│   └── test_data.csv
├── strategy_format/
│   ├── __init__.py
│   └── base_strategy.py
└── trading/
    ├── __init__.py
    ├── analyzer.py
    ├── config_manager.py
    ├── llm_service.py
    ├── scraper.py
    ├── strategy_extractor.py
    ├── strategy_tester.py
    ├── tiktoken_patch.py
    └── validation_service.py
```

### Largest Directories

| Directory | Files | Size |
|-----------|-------|------|
| trading | 9 | 157.8 KB |
| quant_strategy_scraper | 12 | 100.9 KB |
| output/papers | 8 | 79.1 KB |
| trading/paper_selection | 6 | 48.7 KB |
| strategy_format | 2 | 11.3 KB |

## Dependencies

### Module Relationships (Summary)

- `diagnose.py`
- `main.py`
- `output/strategies/llm_strategies/implementations/2302.10175v1_SpatioTemporalMomentumStrategy.py`
- `pipeline_stages.py`
- `trading/paper_selection/__init__.py`
- `trading/paper_selection/integrator.py`
- `trading/paper_selection/paper_selector.py`
- `trading/validation_service.py`

## Documentation

### `analyze_system.py`

Simple System Analyzer - Concise Report Version

### `debug.py`

Crash detector for main.py using faulthandler to identify
the location of segmentation faults

### `diagnose.py`

Diagnostic script to find the exact line causing the segmentation fault

### `main.py`

Main script for the Trading Strategy Paper Scraper with LLM integration
Refactored to use PipelineStages for better organization

### `pipeline_stages.py`

Complete Pipeline Stages module - Refactored from main.py
Organizes the trading strategy paper scraper pipeline into manageable stages

### `strategy_format/base_strategy.py`

Enhanced BaseStrategy class with improved metadata, validation, and error handling
This fully replaces strategy_format/base_strategy.py

### `trading/analyzer.py`

Paper Analysis module for identifying trading strategies in academic papers

### `trading/config_manager.py`

Configuration manager for trading strategy paper scraper

### `trading/llm_service.py`

LLM Service module for extracting trading strategies from papers
With optimizations for token counting, dry run mode, and cost-effective model selection

### `trading/paper_selection/integrator.py`

Integrator module for paper selection

### `trading/scraper.py`

Research Scraper module for finding trading strategies in academic papers

### `trading/strategy_extractor.py`

Strategy Extractor with configurable cost controls
Extracts trading strategies from research papers using LLM

### `trading/strategy_tester.py`

Strategy Tester module for evaluating generated trading strategies - Fixed version

### `trading/tiktoken_patch.py`

Enhanced patch to prevent tiktoken-related segfaults.
This must be imported before any other imports.

### `trading/validation_service.py`

Enhanced Validation Service - Integrated with refactored pipeline
Combines the bulletproof approach of your existing service with BaseStrategy compliance

## Insights & Recommendations

Project structure looks good with no obvious issues.
