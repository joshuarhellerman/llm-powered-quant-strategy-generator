"""
Research Scraper module for finding trading strategies in academic papers
"""

import json
import time
import os
import requests
import logging
import warnings
from datetime import datetime
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import matplotlib.pyplot as plt

# Filter the XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


class ResearchScraper:
    """
    Scrapes research papers from arXiv to find trading strategies
    """

    def __init__(self, query_topics=None, max_papers=50, output_dir="output/papers",
                 trading_keywords=None, arxiv_categories=None, rate_limit=3):
        """
        Initialize the scraper with configuration settings

        Args:
            query_topics (list): List of topics to search for
            max_papers (int): Maximum number of papers to retrieve
            output_dir (str): Directory to save papers
            trading_keywords (list): Keywords to filter relevant papers
            arxiv_categories (list): arXiv categories to filter by
            rate_limit (int): Seconds to wait between API calls
        """
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
        self.arxiv_categories = arxiv_categories or [
            "q-fin.TR",  # Trading and Market Microstructure
            "q-fin.PM",  # Portfolio Management
            "q-fin.ST",  # Statistical Finance
            "q-fin.CP",  # Computational Finance
            "cs.LG"      # Machine Learning
        ]

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
        """
        Search arXiv for papers matching the query

        Args:
            query (str): Search query for arXiv
            max_results (int): Maximum number of results to return

        Returns:
            list: List of paper dictionaries with metadata
        """
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

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()  # Raise exception for HTTP errors
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error searching arXiv: {e}")
            return []

        # Parse the XML response using html.parser which is more forgiving
        soup = BeautifulSoup(response.text, 'html.parser')

        entries = soup.find_all('entry')
        self.logger.info(f"Found {len(entries)} entries for query: '{query}'")

        # Process entries to extract relevant data
        papers = []
        for entry in entries:
            try:
                # Extract paper ID (the last part of the ID URL)
                id_tag = entry.find('id')
                if id_tag is None:
                    continue
                paper_id = id_tag.text.split('/')[-1]

                # Extract title and remove newlines
                title_tag = entry.find('title')
                if title_tag is None:
                    continue
                title = title_tag.text.strip().replace('\n', ' ')

                # Extract abstract and clean it
                summary_tag = entry.find('summary')
                if summary_tag is None:
                    continue
                abstract = summary_tag.text.strip().replace('\n', ' ')

                # Extract authors
                authors = []
                author_tags = entry.find_all('author')
                for author_tag in author_tags:
                    name_tag = author_tag.find('name')
                    if name_tag:
                        authors.append(name_tag.text)

                # Extract published date
                published = None
                published_tag = entry.find('published')
                if published_tag:
                    published = published_tag.text

                # Extract categories/tags
                categories = []
                category_tags = entry.find_all('category')
                for cat_tag in category_tags:
                    term = cat_tag.get('term')
                    if term:
                        categories.append(term)

                # Check if paper is relevant to trading strategies
                if self._is_trading_relevant(title, abstract):
                    paper = {
                        'id': paper_id,
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'published': published,
                        'categories': categories,
                        'url': id_tag.text,
                        'pdf_url': f"https://arxiv.org/pdf/{paper_id}.pdf",
                        'source': 'arxiv'
                    }
                    papers.append(paper)
            except Exception as e:
                self.logger.error(f"Error processing entry: {e}")
                continue

        self.logger.info(f"Found {len(papers)} relevant trading papers for query: '{query}'")
        return papers

    def _is_trading_relevant(self, title, abstract):
        """
        Check if a paper is relevant to trading strategies

        Args:
            title (str): Paper title
            abstract (str): Paper abstract

        Returns:
            bool: True if paper is relevant to trading
        """
        text = (title + " " + abstract).lower()
        return any(keyword.lower() in text for keyword in self.trading_keywords)

    def extract_strategy_details(self, abstract):
        """
        Extract key details about the trading strategy from the abstract

        Args:
            abstract (str): Paper abstract

        Returns:
            dict: Strategy details including type, asset class, and ML technique
        """
        strategy_type = None
        asset_class = None
        ml_technique = None

        # Strategy type detection
        strategy_types = {
            "momentum": ["momentum", "trend following", "breakout"],
            "mean reversion": ["mean reversion", "mean-reversion", "reversal", "oscillation"],
            "arbitrage": ["arbitrage", "spread", "pairs trading", "statistical arbitrage"],
            "market making": ["market making", "market-making", "bid-ask spread", "liquidity"],
            "reinforcement learning": ["reinforcement learning", "q-learning", "rl"],
            "machine learning": ["machine learning", "neural network", "deep learning", "ml"]
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
            "options": ["option", "options", "derivative", "derivatives"],
            "multi-asset": ["portfolio", "multi-asset", "asset allocation"]
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

    def scrape_papers(self, test_limit=None):
        """
        Scrape papers from arXiv based on query topics

        Args:
            test_limit (int, optional): Limit the number of papers for testing

        Returns:
            list: List of paper dictionaries
        """
        all_papers = []

        for query in self.query_topics:
            try:
                # Get papers for this query
                papers = self.search_arxiv(query, max_results=self.max_papers // len(self.query_topics))

                # Check if we got papers back (defensive programming)
                if papers is None:
                    self.logger.warning(f"Search for query '{query}' returned None")
                elif isinstance(papers, list):
                    all_papers.extend(papers)
                else:
                    self.logger.warning(f"Expected list of papers but got {type(papers)} for query '{query}'")

                # Be nice to the API
                self.logger.debug(f"Sleeping for {self.rate_limit} seconds to respect API limits")
                time.sleep(self.rate_limit)

            except Exception as e:
                self.logger.error(f"Error processing query '{query}': {e}")
                # Continue with other queries even if one fails
                continue

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
            filename = os.path.join(self.output_dir, "trading_papers.json")

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
            self.logger.warning(f"File {filename} not found. No papers loaded.")
            # Create a dummy paper for testing in case no file is found
            if limit is not None and limit > 0:
                self.papers = [{
                    'id': 'dummy1',
                    'title': 'Momentum Trading Strategy with Technical Indicators',
                    'abstract': 'This paper explores an advanced momentum trading strategy that combines multiple technical indicators to identify market trends and generate trading signals.',
                    'authors': ['Jane Smith', 'John Doe'],
                    'published': '2023-05-01',
                    'url': 'https://example.com/dummy1',
                    'pdf_url': 'https://example.com/dummy1.pdf',
                    'source': 'arxiv',
                    'categories': ['q-fin.PM', 'q-fin.TR'],
                    'strategy_type': 'momentum',
                    'asset_class': 'equities'
                }]
                self.logger.info("Created 1 dummy paper for testing purposes")
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from {filename}: {e}")

    def save_papers(self, filename=None):
        """
        Save papers to a JSON file

        Args:
            filename (str, optional): Path to save JSON file
        """
        if not self.papers:
            self.logger.warning("No papers to save. Run scrape_papers() first.")
            return

        if filename is None:
            filename = os.path.join(self.output_dir, "trading_papers.json")

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        try:
            with open(filename, 'w') as f:
                json.dump(self.papers, f, indent=2)

            self.logger.info(f"Saved {len(self.papers)} papers to {filename}")

            # Also save individual papers with timestamp to prevent overwriting
            timestamp = datetime.now().strftime("%Y%m%d")
            papers_dir = os.path.join(self.output_dir, f"papers_{timestamp}")
            os.makedirs(papers_dir, exist_ok=True)

            for paper in self.papers:
                paper_filename = os.path.join(papers_dir, f"{paper['id']}.json")
                with open(paper_filename, 'w') as f:
                    json.dump(paper, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving papers to {filename}: {e}")

    def analyze_papers(self):
        """
        Analyze the papers and return statistics

        Returns:
            dict: Statistics about the papers
        """
        if not self.papers:
            self.logger.warning("No papers to analyze. Run scrape_papers() or load_papers() first.")
            return {}

        # Count strategy types
        strategy_counts = {}
        for paper in self.papers:
            strategy_type = paper.get("strategy_type")
            if strategy_type:
                strategy_counts[strategy_type] = strategy_counts.get(strategy_type, 0) + 1
            else:
                strategy_counts["unknown"] = strategy_counts.get("unknown", 0) + 1

        # Count asset classes
        asset_counts = {}
        for paper in self.papers:
            asset_class = paper.get("asset_class")
            if asset_class:
                asset_counts[asset_class] = asset_counts.get(asset_class, 0) + 1
            else:
                asset_counts["unknown"] = asset_counts.get("unknown", 0) + 1

        # Count ML techniques
        ml_counts = {}
        for paper in self.papers:
            ml_technique = paper.get("ml_technique")
            if ml_technique:
                ml_counts[ml_technique] = ml_counts.get(ml_technique, 0) + 1

        # Calculate popularity over time if dates are available
        date_counts = {}
        for paper in self.papers:
            if 'published' in paper and paper['published']:
                try:
                    year = paper['published'][:4]  # Extract year from date
                    if year:
                        date_counts[year] = date_counts.get(year, 0) + 1
                except Exception as e:
                    self.logger.debug(f"Error processing date for paper {paper.get('id')}: {e}")

        # Create visualization
        if self.papers:
            self._create_analysis_plots(strategy_counts, asset_counts, ml_counts, date_counts)

        # Return the analysis
        return {
            "total_papers": len(self.papers),
            "strategy_counts": strategy_counts,
            "asset_counts": asset_counts,
            "ml_technique_counts": ml_counts,
            "date_distribution": date_counts
        }

    def _create_analysis_plots(self, strategy_counts, asset_counts, ml_counts, date_counts):
        """
        Create analysis plots based on paper statistics

        Args:
            strategy_counts (dict): Strategy type counts
            asset_counts (dict): Asset class counts
            ml_counts (dict): ML technique counts
            date_counts (dict): Publication year counts
        """
        try:
            plt.figure(figsize=(15, 15))
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Plot strategy types
            self._plot_bar_chart(
                axes[0, 0],
                strategy_counts,
                "Papers by Strategy Type",
                "Strategy Type",
                "Count"
            )

            # Plot asset classes
            self._plot_bar_chart(
                axes[0, 1],
                asset_counts,
                "Papers by Asset Class",
                "Asset Class",
                "Count"
            )

            # Plot ML techniques
            self._plot_bar_chart(
                axes[1, 0],
                ml_counts,
                "Papers by ML Technique",
                "ML Technique",
                "Count"
            )

            # Plot date distribution
            if date_counts:
                # Convert to lists for safer plotting
                dates_sorted = dict(sorted(date_counts.items()))
                date_keys = list(dates_sorted.keys())
                date_values = list(dates_sorted.values())

                # Check if we have valid data to plot
                if date_keys and date_values and None not in date_keys:
                    axes[1, 1].bar(date_keys, date_values, color='skyblue')
                    axes[1, 1].set_title("Papers by Year", fontsize=14)
                    axes[1, 1].set_xlabel("Year", fontsize=12)
                    axes[1, 1].set_ylabel("Count", fontsize=12)
                    axes[1, 1].tick_params(axis='x', rotation=45)
                    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
                else:
                    axes[1, 1].text(0.5, 0.5, 'No publication date data available',
                                    horizontalalignment='center', verticalalignment='center',
                                    transform=axes[1, 1].transAxes)
            else:
                axes[1, 1].text(0.5, 0.5, 'No publication date data available',
                                horizontalalignment='center', verticalalignment='center',
                                transform=axes[1, 1].transAxes)

            plt.tight_layout()

            # Create output directory if it doesn't exist
            plots_dir = os.path.join(self.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(plots_dir, f"paper_analysis_{timestamp}.png"), dpi=300)
            plt.savefig(os.path.join(self.output_dir, "paper_analysis.png"), dpi=300)
            plt.close()

            self.logger.info(f"Saved paper analysis visualization to {plots_dir}")

        except Exception as e:
            self.logger.error(f"Error creating analysis plots: {e}")

    def _plot_bar_chart(self, ax, data, title, xlabel, ylabel):
        """
        Helper method to plot a bar chart

        Args:
            ax (matplotlib.axes): Axes to plot on
            data (dict): Data to plot
            title (str): Chart title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
        """
        if not data:
            ax.text(0.5, 0.5, f'No {xlabel.lower()} data available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            return

        # Convert to lists for plotting
        keys = list(data.keys())
        values = list(data.values())

        # Check if we have valid data
        if not keys or None in keys:
            ax.text(0.5, 0.5, f'No {xlabel.lower()} data available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            return

        # Plot the data
        bars = ax.bar(keys, values, color='lightblue')

        # Customize the plot
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')