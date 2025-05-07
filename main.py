#!/usr/bin/env python
"""
Trading Strategy Paper Scraper and Code Generator

This script scrapes academic papers about trading strategies from arXiv
and converts them into executable Python code.
"""

import argparse
import os
import logging
from trading.scraper import ResearchScraper
from trading.generator import StrategyGenerator
from trading.config_manager import ConfigManager


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Trading Strategy Paper Scraper and Code Generator"
    )

    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--query-topics", "-q",
        nargs="+",
        default=None,
        help="Topics to search for in research papers (e.g., 'momentum trading' 'reinforcement learning')"
    )

    parser.add_argument(
        "--max-papers", "-m",
        type=int,
        default=None,
        help="Maximum number of papers to scrape"
    )

    parser.add_argument(
        "--no-scrape", "-n",
        action="store_true",
        help="Skip scraping and use existing papers"
    )

    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for results"
    )

    parser.add_argument(
        "--arxiv-categories", "-a",
        nargs="+",
        default=None,
        help="ArXiv categories to search (e.g., 'q-fin.PM' 'cs.LG')"
    )

    parser.add_argument(
        "--test-mode", "-t",
        action="store_true",
        help="Run in test mode with limited papers"
    )

    parser.add_argument(
        "--test-limit", "-l",
        type=int,
        default=1,
        help="Number of papers to process in test mode"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config_manager = ConfigManager(args.config)

    # Override config with command line arguments if provided
    config_updates = {}

    if args.query_topics:
        config_updates["scraping"] = config_updates.get("scraping", {})
        config_updates["scraping"]["query_topics"] = args.query_topics

    if args.max_papers:
        config_updates["scraping"] = config_updates.get("scraping", {})
        config_updates["scraping"]["max_papers"] = args.max_papers

    if args.output_dir:
        config_updates["output"] = config_updates.get("output", {})
        config_updates["output"]["base_dir"] = args.output_dir

    if args.arxiv_categories:
        config_updates["scraping"] = config_updates.get("scraping", {})
        config_updates["scraping"]["arxiv_categories"] = args.arxiv_categories

    if args.test_mode:
        config_updates["scraping"] = config_updates.get("scraping", {})
        config_updates["scraping"]["test_mode"] = True
        config_updates["scraping"]["test_paper_limit"] = args.test_limit

    if config_updates:
        config_manager.update_config(config_updates)

    # Get configuration
    scraping_config = config_manager.get_scraping_config()
    generator_config = config_manager.get_generator_config()
    output_dirs = config_manager.get_output_dirs()

    # Create output directories
    papers_dir = output_dirs["papers_dir"]
    strategies_dir = output_dirs["strategies_dir"]
    visualizations_dir = output_dirs["visualizations_dir"]

    for directory in output_dirs.values():
        os.makedirs(directory, exist_ok=True)

    # Print banner
    print("\n" + "=" * 80)
    print(" Trading Strategy Paper Scraper and Code Generator ")
    print("=" * 80)
    print(f"Configuration File: {args.config}")
    print(f"Query Topics: {scraping_config['query_topics']}")
    print(f"Max Papers: {scraping_config['max_papers']}")
    print(f"Output Directory: {output_dirs['base_dir']}")
    print(f"Scrape New: {not args.no_scrape}")

    # Check if we're in test mode
    test_mode = scraping_config.get("test_mode", False)
    test_limit = scraping_config.get("test_paper_limit", 1)

    if test_mode:
        print(f"⚠️ TEST MODE: Limited to {test_limit} paper(s)")

    if 'arxiv_categories' in scraping_config:
        print(f"ArXiv Categories: {scraping_config['arxiv_categories']}")

    print("=" * 80 + "\n")

    # Step 1: Scrape research papers
    scraper = ResearchScraper(
        query_topics=scraping_config["query_topics"],
        max_papers=scraping_config["max_papers"],
        output_dir=papers_dir,
        trading_keywords=scraping_config.get("trading_keywords"),
        arxiv_categories=scraping_config.get("arxiv_categories"),
        rate_limit=scraping_config.get("rate_limit", 3)
    )

    if not args.no_scrape:
        print("Step 1: Scraping research papers from arXiv...")

        # Check if we're in test mode for scraping
        if test_mode:
            papers = scraper.scrape_papers(test_limit=test_limit)
        else:
            papers = scraper.scrape_papers()
    else:
        print("Step 1: Loading existing papers...")

        # Check if we're in test mode for loading as well
        if test_mode:
            scraper.load_papers(f"{papers_dir}/trading_papers.json", limit=test_limit)
        else:
            scraper.load_papers(f"{papers_dir}/trading_papers.json")

        papers = scraper.papers

    # Analyze the papers
    print("\nPaper Analysis:")
    analysis = scraper.analyze_papers()

    # Print analysis results
    print("\nFound Papers by Strategy Type:")
    for strategy_type, count in analysis["strategy_counts"].items():
        print(f" - {strategy_type}: {count}")

    # Step 2: Generate strategy code
    print("\nStep 2: Generating trading strategy code...")
    generator = StrategyGenerator(
        papers=papers,
        output_dir=strategies_dir,
        config=generator_config
    )
    strategies = generator.generate_strategies()

    # Print summary
    print("\n" + "=" * 80)
    print(" Summary ")
    print("=" * 80)
    print(f"Papers Found: {len(papers)}")
    print(f"Strategies Generated: {len(strategies)}")
    print("=" * 80)
    print(f"\nAll papers saved to: {papers_dir}/")
    print(f"All strategy code saved to: {strategies_dir}/")
    print(f"Visualizations saved to: {visualizations_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()