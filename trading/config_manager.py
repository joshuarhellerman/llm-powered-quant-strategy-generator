"""
Configuration manager for trading strategy paper scraper
"""

import os
import yaml
import logging


class ConfigManager:
    """
    Configuration manager that loads and validates configuration from YAML files
    """

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()

        # Ensure all required config sections exist
        self.update_config_defaults()

    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Validate config
            self._validate_config(config)
            return config
        except FileNotFoundError:
            print(f"Configuration file not found at {self.config_path}")
            print("Creating default configuration file...")
            self._create_default_config()
            return self._load_config()
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            return self._get_default_config()

    def _validate_config(self, config):
        """Validate configuration structure"""
        required_sections = ["scraping", "analysis", "generator", "output"]

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Check required fields
        if "max_papers" not in config["scraping"]:
            raise ValueError("Missing required field: scraping.max_papers")

        if "query_topics" not in config["scraping"]:
            raise ValueError("Missing required field: scraping.query_topics")

    def _create_default_config(self):
        """Create default configuration file"""
        default_config = self._get_default_config()

        os.makedirs(os.path.dirname(self.config_path) or '.', exist_ok=True)

        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        print(f"Default configuration created at {self.config_path}")

    def _get_default_config(self):
        """Get default configuration"""
        return {
            "scraping": {
                "max_papers": 50,
                "rate_limit": 3,
                "query_topics": [
                    "quantitative trading strategies",
                    "algorithmic trading",
                    "momentum trading strategy",
                    "mean reversion trading",
                    "reinforcement learning trading",
                    "machine learning trading strategy"
                ],
                "trading_keywords": [
                    "trading", "strategy", "algorithmic", "quantitative", "momentum",
                    "mean-reversion", "statistical arbitrage", "market making",
                    "high-frequency", "portfolio optimization", "factor", "risk",
                    "alpha", "machine learning", "reinforcement learning"
                ],
                "arxiv_categories": [
                    "q-fin.PM", "q-fin.TR", "q-fin.ST", "cs.LG", "stat.ML"
                ]
            },
            "analysis": {
                "strategy_threshold": 2,
                "top_recommendations": 5
            },
            "generator": {
                "parameters": {
                    "lookback_periods": [5, 10, 20, 50, 100],
                    "buy_thresholds": [0.01, 0.02, 0.03, 0.04, 0.05],
                    "sell_thresholds": [-0.05, -0.04, -0.03, -0.02, -0.01],
                    "thresholds": [1.0, 1.5, 2.0, 2.5]
                },
                "template_weights": {
                    "momentum": 1.0,
                    "mean_reversion": 1.0,
                    "reinforcement_learning": 0.8,
                    "transformer": 0.7
                }
            },
            "output": {
                "base_dir": "output",
                "papers_dir": "papers",
                "strategies_dir": "strategies",
                "visualizations_dir": "visualizations",
                "log_level": "INFO"
            },
            "test_mode": {
                "enabled": False,
                "ensure_success": True,
                "lower_thresholds": True,
                "max_attempts": 3,
                "papers_per_attempt": 3,
                "synthetic_fallback": True,
                "limit": 3
            },
            "paper_selection": {
                "output_dir": "output/papers/selected",
                "paper_selector": {
                    "basic_threshold": 0.3,
                    "semantic_threshold": 0.6,
                    "final_threshold": 0.7,
                    "semantic_batch_size": 10
                },
                "test_thresholds": {
                    "basic_threshold": 0.1,
                    "semantic_threshold": 0.2,
                    "final_threshold": 0.3
                },
                "max_papers": 50
            },
            "llm": {
                "primary_provider": "claude",
                "claude_model": "claude-3-5-haiku-20241022",
                "cache_dir": "output/llm_cache",
                "dry_run": False
            }
        }

    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_level_str = self.config["output"].get("log_level", "INFO")
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def get_scraping_config(self):
        """Get scraping configuration"""
        return self.config["scraping"]

    def get_analysis_config(self):
        """Get analysis configuration"""
        return self.config["analysis"]

    def get_generator_config(self):
        """Get generator configuration"""
        return self.config["generator"]

    def get_output_config(self):
        """Get output configuration"""
        return self.config["output"]

    def get_test_config(self):
        """Get test mode configuration"""
        test_config = self.config.get('test_mode', {})
        if not test_config:
            # Create default test configuration
            test_config = {
                'enabled': False,
                'ensure_success': True,
                'lower_thresholds': True,
                'max_attempts': 3,
                'papers_per_attempt': 3,
                'synthetic_fallback': True,
                'limit': 3
            }
            self.config['test_mode'] = test_config
        return test_config

    def get_output_dirs(self):
        """Get output directories"""
        output_config = self.get_output_config()
        base_dir = output_config["base_dir"]

        return {
            "base_dir": base_dir,
            "papers_dir": os.path.join(base_dir, output_config["papers_dir"]),
            "strategies_dir": os.path.join(base_dir, output_config["strategies_dir"]),
            "visualizations_dir": os.path.join(base_dir, output_config["visualizations_dir"])
        }

    def update_config(self, config_updates):
        """Update configuration with new values"""
        def _update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    _update_dict(d[k], v)
                else:
                    d[k] = v

        _update_dict(self.config, config_updates)

        # Save updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        return self.config

    def update_config_defaults(self):
        """Ensure all required config sections exist with defaults"""
        # Ensure test_mode section exists
        if 'test_mode' not in self.config:
            self.config['test_mode'] = {
                'enabled': False,
                'ensure_success': True,
                'lower_thresholds': True,
                'max_attempts': 3,
                'papers_per_attempt': 3,
                'synthetic_fallback': True,
                'limit': 3
            }

        # Ensure paper_selection section exists
        if 'paper_selection' not in self.config:
            self.config['paper_selection'] = {
                'output_dir': 'output/papers/selected',
                'paper_selector': {
                    'basic_threshold': 0.3,
                    'semantic_threshold': 0.6,
                    'final_threshold': 0.7,
                    'semantic_batch_size': 10
                },
                'test_thresholds': {
                    'basic_threshold': 0.1,
                    'semantic_threshold': 0.2,
                    'final_threshold': 0.3
                },
                'max_papers': 50
            }

        # Ensure llm section exists
        if 'llm' not in self.config:
            self.config['llm'] = {
                'primary_provider': 'claude',
                'claude_model': 'claude-3-5-haiku-20241022',
                'cache_dir': 'output/llm_cache',
                'dry_run': False
            }