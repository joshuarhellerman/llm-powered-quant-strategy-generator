"""
Research Scraper module for finding trading strategies in academic papers
"""

import json
import time
import os
import requests
import logging
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


class ResearchScraper:
    """
    Scrapes research papers from arXiv to find trading strategies
    """

    def __init__(self, query_topics=None, max_papers=50, output_dir="output/papers",
                 trading_keywords=None, arxiv_categories=None, rate_limit=3):
        self.query_topics = query_topics or [
            "quantitative trading strategies",
            "algorithmic trading",
            "momentum trading strategy",
            "mean reversion trading",
            "reinforcement learning trading",
            "machine learning trading strategy"
        ]
        self.max_papers = max_papers
        self.papers = []
        self.output_dir = output_dir
        self.rate_limit = rate_limit
        self.arxiv_categories = arxiv_categories

        self.trading_keywords = trading_keywords or [
            "trading", "strategy", "algorithmic", "quantitative", "momentum",
            "mean-reversion", "statistical arbitrage", "market making",
            "high-frequency", "portfolio optimization", "factor", "risk",
            "alpha", "machine learning", "reinforcement learning"
        ]

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def search_arxiv(self, query, max_results=50):
        """Search arXiv for papers matching the query"""
        base_url = "https://export.arxiv.org/api/query"

        # Build search query
        search_query = f"all:{query}"

        # Add categories if specified
        if self.arxiv_categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in self.arxiv_categories])
            search_query = f"({search_query}) AND ({cat_query})"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        self.logger.info(f"Searching arXiv for: '{query}'...")
        self.logger.debug(f"Full search query: {search_query}")

        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            self.logger.error(f"Error searching arXiv: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, 'lxml')
        entries = soup.find_all('entry')

        self.logger.info(f"Found {len(entries)} entries for query: '{query}'")

        results = []
        for entry in entries:
            title = entry.find('title').text.strip()
            abstract = entry.find('summary').text.strip()
            published = entry.find('published').text.strip()
            link = entry.find('link', {'rel': 'alternate'}).get('href') if entry.find('link',
                                                                                      {'rel': 'alternate'}) else None

            if self._is_trading_relevant(title, abstract):
                paper_id = entry.find('id').text.strip().split('/')[-1]

                results.append({
                    'id': paper_id,
                    'title': title,
                    'abstract': abstract,
                    'published': published,
                    'link': link,
                    'source': 'arxiv'
                })

        self.logger.info(f"Found {len(results)} relevant trading papers for query: '{query}'")
        return results

    def _is_trading_relevant(self, title, abstract):
        """Check if a paper is relevant to trading strategies"""
        text = (title + " " + abstract).lower()
        return any(keyword in text for keyword in self.trading_keywords)

    def extract_strategy_details(self, abstract):
        """
        Extract key details about the trading strategy from the abstract
        """
        strategy_type = None
        asset_class = None
        ml_technique = None

        # Strategy type detection
        strategy_types = {
            "momentum": ["momentum", "trend following", "breakout"],
            "mean reversion": ["mean reversion", "mean-reversion", "reversal", "oscillation"],
            "arbitrage": ["arbitrage", "spread", "pairs trading", "statistical arbitrage"],
            "market making": ["market making", "market-making", "bid-ask spread", "liquidity"]
        }

        for stype, keywords in strategy_types.items():
            if any(keyword in abstract.lower() for keyword in keywords):
                strategy_type = stype
                break

        # Asset class detection
        asset_classes = {
            "equities": ["stock", "equity", "equities", "share"],
            "futures": ["futures", "commodity", "commodities"],
            "forex": ["forex", "fx", "currency", "currencies"],
            "crypto": ["crypto", "cryptocurrency", "bitcoin", "altcoin", "token"],
            "options": ["option", "options", "derivative", "derivatives"]
        }

        for aclass, keywords in asset_classes.items():
            if any(keyword in abstract.lower() for keyword in keywords):
                asset_class = aclass
                break

        # ML technique detection
        ml_techniques = {
            "deep learning": ["deep learning", "neural network", "cnn", "rnn", "lstm"],
            "reinforcement learning": ["reinforcement learning", "q-learning", "dqn", "a2c", "ppo", "ddpg"],
            "transformer": ["transformer", "attention", "bert", "gpt"],
            "random forest": ["random forest", "decision tree", "ensemble"],
            "svm": ["support vector", "svm"],
            "genetic algorithm": ["genetic algorithm", "evolutionary", "genetic programming"]
        }

        for technique, keywords in ml_techniques.items():
            if any(keyword in abstract.lower() for keyword in keywords):
                ml_technique = technique
                break

        return {
            "strategy_type": strategy_type,
            "asset_class": asset_class,
            "ml_technique": ml_technique
        }

    # In the ResearchScraper class, update these methods:

    def scrape_papers(self, test_limit=None):
        """
        Scrape papers from arXiv based on query topics

        Args:
            test_limit (int, optional): Limit the number of papers for testing
        """
        all_papers = []

        for query in self.query_topics:
            papers = self.search_arxiv(query, max_results=self.max_papers // len(self.query_topics))
            all_papers.extend(papers)

            # Be nice to the API
            self.logger.debug(f"Sleeping for {self.rate_limit} seconds to respect API limits")
            time.sleep(self.rate_limit)

        # Remove duplicates
        unique_papers = []
        seen_ids = set()

        for paper in all_papers:
            if paper['id'] not in seen_ids:
                seen_ids.add(paper['id'])

                # Extract strategy details
                details = self.extract_strategy_details(paper['abstract'])
                paper.update(details)

                unique_papers.append(paper)

        # Apply test limit if specified
        if test_limit is not None and test_limit > 0:
            self.logger.info(f"Test mode: Limiting to {test_limit} paper(s)")
            unique_papers = unique_papers[:test_limit]

        self.papers = unique_papers
        self.logger.info(f"Found {len(self.papers)} unique relevant papers across all queries")

        # Save the papers to file
        self.save_papers()

        return self.papers

    def load_papers(self, filename=None, limit=None):
        """
        Load papers from a JSON file

        Args:
            filename (str, optional): Path to JSON file
            limit (int, optional): Limit the number of papers for testing
        """
        if filename is None:
            filename = f"{self.output_dir}/trading_papers.json"

        try:
            with open(filename, 'r') as f:
                papers = json.load(f)

            # Apply test limit if specified
            if limit is not None and limit > 0:
                self.logger.info(f"Test mode: Limiting to {limit} paper(s)")
                self.papers = papers[:limit]
            else:
                self.papers = papers

            self.logger.info(f"Loaded {len(self.papers)} papers from {filename}")
        except FileNotFoundError:
            self.logger.warning(f"File {filename} not found.")

    def save_papers(self, filename=None):
        """Save papers to a JSON file"""
        if not self.papers:
            self.logger.warning("No papers to save. Run scrape_papers() first.")
            return

        if filename is None:
            filename = f"{self.output_dir}/trading_papers.json"

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(self.papers, f, indent=2)

        self.logger.info(f"Saved {len(self.papers)} papers to {filename}")

        # Also save individual papers
        for paper in self.papers:
            paper_filename = f"{self.output_dir}/{paper['id']}.json"
            with open(paper_filename, 'w') as f:
                json.dump(paper, f, indent=2)

    def analyze_papers(self):
        """Analyze the papers and return statistics"""
        # Count strategy types
        strategy_counts = {}
        for paper in self.papers:
            strategy_type = paper.get("strategy_type", "unknown")
            if strategy_type and strategy_type != "None":
                strategy_counts[strategy_type] = strategy_counts.get(strategy_type, 0) + 1
            else:
                strategy_counts["unknown"] = strategy_counts.get("unknown", 0) + 1

        # Count asset classes
        asset_counts = {}
        for paper in self.papers:
            asset_class = paper.get("asset_class", "unknown")
            if asset_class and asset_class != "None":
                asset_counts[asset_class] = asset_counts.get(asset_class, 0) + 1
            else:
                asset_counts["unknown"] = asset_counts.get("unknown", 0) + 1

        # Calculate popularity over time if dates are available
        dates = {}
        for paper in self.papers:
            if 'published' in paper and paper['published']:
                try:
                    year = paper['published'][:4]  # Extract year from date
                    if year and year != "None":
                        dates[year] = dates.get(year, 0) + 1
                except:
                    # Skip if error processing the date
                    pass

        # Create visualization
        if self.papers:  # Only create visualization if we have papers
            plt.figure(figsize=(15, 10))
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Plot strategy types
            if strategy_counts:
                # Convert to lists for safer plotting
                strategies = list(strategy_counts.keys())
                strategy_values = list(strategy_counts.values())

                # Check if we have valid data to plot
                if strategies and strategy_values and None not in strategies:
                    axes[0, 0].bar(strategies, strategy_values)
                    axes[0, 0].set_title("Papers by Strategy Type")
                    axes[0, 0].set_xlabel("Strategy Type")
                    axes[0, 0].set_ylabel("Count")
                    axes[0, 0].tick_params(axis='x', rotation=45)
                else:
                    axes[0, 0].text(0.5, 0.5, 'No strategy data available',
                                    horizontalalignment='center', verticalalignment='center',
                                    transform=axes[0, 0].transAxes)

            # Plot asset classes
            if asset_counts:
                # Convert to lists for safer plotting
                assets = list(asset_counts.keys())
                asset_values = list(asset_counts.values())

                # Check if we have valid data to plot
                if assets and asset_values and None not in assets:
                    axes[0, 1].bar(assets, asset_values)
                    axes[0, 1].set_title("Papers by Asset Class")
                    axes[0, 1].set_xlabel("Asset Class")
                    axes[0, 1].set_ylabel("Count")
                    axes[0, 1].tick_params(axis='x', rotation=45)
                else:
                    axes[0, 1].text(0.5, 0.5, 'No asset class data available',
                                    horizontalalignment='center', verticalalignment='center',
                                    transform=axes[0, 1].transAxes)

            # Plot date distribution if available
            if dates:
                # Convert to lists for safer plotting
                dates_sorted = dict(sorted(dates.items()))
                date_keys = list(dates_sorted.keys())
                date_values = list(dates_sorted.values())

                # Check if we have valid data to plot
                if date_keys and date_values and None not in date_keys:
                    axes[1, 0].bar(date_keys, date_values)
                    axes[1, 0].set_title("Papers by Year")
                    axes[1, 0].set_xlabel("Year")
                    axes[1, 0].set_ylabel("Count")
                    axes[1, 0].tick_params(axis='x', rotation=45)
                else:
                    axes[1, 0].text(0.5, 0.5, 'No publication date data available',
                                    horizontalalignment='center', verticalalignment='center',
                                    transform=axes[1, 0].transAxes)
            else:
                axes[1, 0].text(0.5, 0.5, 'No publication date data available',
                                horizontalalignment='center', verticalalignment='center',
                                transform=axes[1, 0].transAxes)

            # Placeholder for other analyses
            axes[1, 1].text(0.5, 0.5, 'Future Analysis',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[1, 1].transAxes)

            plt.tight_layout()

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(f"{self.output_dir}/paper_analysis.png"), exist_ok=True)

            # Save the figure
            plt.savefig(f"{self.output_dir}/paper_analysis.png")
            plt.close()

        # Return the analysis
        return {
            "total_papers": len(self.papers),
            "strategy_counts": strategy_counts,
            "asset_counts": asset_counts,
            "date_distribution": dates
        }