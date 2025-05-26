"""
LLM Service module for extracting trading strategies from papers
With optimizations for token counting, dry run mode, and cost-effective model selection
"""

import os
import json
import time
import logging
import requests
import re
import importlib.util
from typing import Dict, List, Any, Optional, Union

# This helps avoid issues with importing tiktoken
TIKTOKEN_AVAILABLE = False
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    pass
except Exception as e:
    print(f"Error during tiktoken import: {e}")
    pass

class LLMService:
    """
    Service for using LLMs to extract trading strategies from research papers
    """

    def __init__(self, config=None):
        """
        Initialize the LLM Service

        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configure providers
        self.primary_provider = self.config.get("primary_provider", "claude")
        self.fallback_providers = self.config.get("fallback_providers", ["llama3"])

        # Dry run mode - simulate API calls without making them
        self.dry_run = self.config.get("dry_run", False)
        if self.dry_run:
            self.logger.info("🔍 DRY RUN MODE ENABLED - No API calls will be made")

        # Set up the Claude model - use Haiku by default for cost efficiency
        # Get the model from config, or use a valid default
        self.claude_model = self.config.get("claude_model", "claude-3-5-haiku-20241022")
        self.logger.info(f"Using Claude model: {self.claude_model}")

        # Token and cost estimation parameters
        self.model_costs = {
            "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},  # $1.00/$5.00 per million tokens
            "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},  # $15.00/$75.00 per million tokens
            "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00}  # $3.00/$15.00 per million tokens
        }

        # Initialize API keys
        self.api_keys = {}
        self._setup_api_keys()

        # Setup clients
        self.clients = {}
        self._setup_clients()

        # Setup cache
        self.cache_dir = self.config.get("cache_dir", "cache/llm_responses")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Rate limiting configuration
        self.rate_limits = {
            "claude": {"requests_per_minute": 10, "last_request_time": 0},
            "llama3": {"requests_per_minute": 20, "last_request_time": 0}
        }

        # Initialize token counters for spend tracking
        self.token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0.0
        }

        # Set tiktoken availability flag
        self.tiktoken_available = TIKTOKEN_AVAILABLE

    def _apply_rate_limiting(self, provider="claude"):
        """
        Apply rate limiting between API requests

        Args:
            provider (str): The provider to apply rate limiting for
        """
        if provider not in self.rate_limits:
            return

        current_time = time.time()
        last_request_time = self.rate_limits[provider]["last_request_time"]
        requests_per_minute = self.rate_limits[provider]["requests_per_minute"]

        # Calculate minimum time between requests (in seconds)
        min_interval = 60.0 / requests_per_minute

        # Calculate how long since last request
        time_since_last = current_time - last_request_time

        # If we need to wait, do so
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            self.logger.debug(f"Rate limiting {provider}: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        # Update the last request time
        self.rate_limits[provider]["last_request_time"] = time.time()

    def _setup_api_keys(self):
        """Setup API keys from various sources - SIMPLIFIED VERSION"""
        # Try to get Claude API key from various sources
        claude_key = None

        # 1. Try from environment variable FIRST (most reliable)
        if os.environ.get("ANTHROPIC_API_KEY"):
            claude_key = os.environ.get("ANTHROPIC_API_KEY")
            self.logger.info("Found Claude API key in environment variable ANTHROPIC_API_KEY")

        # 2. Try direct claude_api_key in config
        elif self.config.get("claude_api_key"):
            claude_key = self.config.get("claude_api_key")
            self.logger.info("Found Claude API key in config.claude_api_key")

        # 3. Try from api_keys dictionary in config
        elif self.config.get("api_keys", {}).get("claude"):
            claude_key = self.config.get("api_keys", {}).get("claude")
            self.logger.info("Found Claude API key in config.api_keys.claude")

        # Validate and store the key
        if claude_key and self._validate_api_key(claude_key, "claude"):
            self.api_keys["claude"] = claude_key
            self.logger.info(f"✅ Claude API key validated and stored: {claude_key[:8]}...{claude_key[-4:]}")
        else:
            self.logger.warning("❌ No valid Claude API key found")

    def _validate_api_key(self, key, provider="claude"):
        """
        Validate that an API key looks reasonable

        Args:
            key (str): API key to validate
            provider (str): Provider name (claude, etc.)

        Returns:
            bool: Whether the key passes basic validation
        """
        if not key:
            self.logger.error(f"No {provider} API key provided")
            return False

        if len(key) < 10:  # API keys are typically much longer
            self.logger.error(f"{provider} API key is suspiciously short: {len(key)} chars")
            return False

        # Anthropic API keys typically start with "sk-ant-"
        if provider == "claude" and not key.startswith("sk-ant-"):
            self.logger.warning(f"Claude API key doesn't have expected prefix 'sk-ant-' but proceeding anyway")

        return True

    def _setup_clients(self):
        """Setup LLM clients based on available providers - SIMPLIFIED VERSION"""
        # Skip client setup if in dry run mode
        if self.dry_run:
            self.logger.info("Dry run mode: Skipping actual client initialization")
            return

        # Get API key directly from environment or stored keys
        api_key = os.environ.get("ANTHROPIC_API_KEY") or self.api_keys.get("claude")

        if api_key:
            try:
                self.logger.info(f"Initializing Claude client with API key: {api_key[:8]}...{api_key[-4:]}")

                # Import and initialize anthropic client
                import anthropic
                self.clients["claude"] = anthropic.Anthropic(api_key=api_key)

                # Test the client with a simple call
                try:
                    test_response = self.clients["claude"].messages.create(
                        model="claude-3-5-haiku-20241022",
                        max_tokens=10,
                        messages=[{"role": "user", "content": "test"}]
                    )
                    self.logger.info("✅ Claude client initialized and tested successfully")
                except Exception as test_e:
                    self.logger.error(f"Claude client test failed: {test_e}")
                    # Keep the client anyway, maybe the test failed for other reasons

            except Exception as e:
                self.logger.error(f"Failed to initialize Claude client: {e}")
        else:
            self.logger.error("❌ No Claude API key available for client initialization")

        # Log final status
        if "claude" in self.clients:
            self.logger.info("✅ Claude client is ready for use")
        else:
            self.logger.error("❌ Claude client not available")

    def _count_tokens(self, text):
        """
        Count tokens safely with fallback to estimation

        Args:
            text (str): Text to count tokens for

        Returns:
            int: Number of tokens
        """
        # Handle None case
        if text is None:
            return 0

        # Ensure text is a string
        if not isinstance(text, str):
            try:
                text = str(text)
            except:
                return 0

        # Empty string case
        if not text:
            return 0

        # Try to use tiktoken if available
        if self.tiktoken_available:
            try:
                # Import and create a new tokenizer each time
                # This prevents memory issues and segfaults from persistent objects
                import tiktoken
                tokenizer = tiktoken.get_encoding("cl100k_base")
                tokens = tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                # Fall through to estimation
                pass

        # Estimation (4 chars per token for English text is a reasonable approximation)
        return max(1, len(text) // 4)

    def _estimate_cost(self, input_tokens, output_tokens, model=None):
        """
        Estimate the cost of an API call based on token counts

        Args:
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
            model (str): Model name to use for pricing

        Returns:
            float: Estimated cost in USD
        """
        model = model or self.claude_model

        # Get costs per million tokens
        if model in self.model_costs:
            costs = self.model_costs[model]
        else:
            # Default to Haiku pricing if model not found
            costs = self.model_costs["claude-3-5-haiku-20241022"]

        input_cost = (input_tokens / 1000000) * costs["input"]
        output_cost = (output_tokens / 1000000) * costs["output"]

        return input_cost + output_cost

    def _log_token_usage(self, input_tokens, output_tokens, model=None):
        """
        Log token usage and update running totals

        Args:
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
            model (str): Model used
        """
        model = model or self.claude_model
        estimated_cost = self._estimate_cost(input_tokens, output_tokens, model)

        # Update running totals
        self.token_usage["input_tokens"] += input_tokens
        self.token_usage["output_tokens"] += output_tokens
        self.token_usage["estimated_cost"] += estimated_cost

        # Log usage
        self.logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}")
        self.logger.info(f"Estimated cost: ${estimated_cost:.4f} (Total: ${self.token_usage['estimated_cost']:.4f})")

    def _get_cache_key(self, prompt, provider=None):
        """Generate a cache key for a prompt"""
        import hashlib

        # Create hash of the prompt text
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        # Include provider in the key if specified
        if provider:
            return f"{provider}_{prompt_hash}"

        return prompt_hash

    def _check_cache(self, cache_key):
        """Check if a response is cached for the given key"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache file {cache_file}: {e}")

        return None

    def _save_to_cache(self, cache_key, response):
        """Save a response to the cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        try:
            with open(cache_file, 'w') as f:
                json.dump(response, f)
        except Exception as e:
            self.logger.warning(f"Failed to save to cache file {cache_file}: {e}")

    def _get_mock_response(self, prompt_type="general"):
        """
        Generate a mock response for dry run mode

        Args:
            prompt_type (str): Type of prompt to generate mock response for

        Returns:
            str: A mock response
        """
        if prompt_type == "strategy_extraction":
            return """
            ```json
            {
              "strategy_name": "Spatio-Temporal Momentum Strategy",
              "core_mechanism": "This strategy combines time-series momentum with cross-sectional ranking to identify assets with persistent price trends that are likely to continue.",
              "indicators": [
                {
                  "name": "Time-Series Momentum",
                  "definition": "12-month price momentum with 1-month lag",
                  "parameters": [12, 1]
                },
                {
                  "name": "Cross-Sectional Rank",
                  "definition": "Relative ranking of momentum within asset universe",
                  "parameters": ["quintiles"]
                }
              ],
              "key_formulas": [
                {
                  "description": "Time-series momentum calculation",
                  "latex": "MOM_{i,t} = \\frac{P_{i,t-1}}{P_{i,t-13}} - 1",
                  "python_equivalent": "momentum = data['close'].shift(1) / data['close'].shift(13) - 1"
                }
              ],
              "asset_classes": ["Stocks", "ETFs"],
              "market_conditions": ["Bull markets", "Trending environments"],
              "time_frames": ["Monthly"],
              "risk_management": {
                "position_sizing": "Equal weight within quintiles",
                "stop_loss": "Not specified",
                "risk_limits": "Long-short portfolio construction"
              }
            }
            ```
            """
        elif prompt_type == "implementation":
            return """
            ```python
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'strategy_format'))
            from base_strategy import BaseStrategy
            import pandas as pd
            import numpy as np


            class SpatioTemporalMomentumStrategy(BaseStrategy):
                \"\"\"
                Spatio-Temporal Momentum Strategy
                
                Based on: Spatio-Temporal Momentum: Jointly Learning Time-Series and Cross-Sectional Strategies
                \"\"\"

                def _initialize_parameters(self, params):
                    \"\"\"Initialize strategy parameters\"\"\"
                    self.momentum_lookback = params.get('momentum_lookback', 12)
                    self.momentum_lag = params.get('momentum_lag', 1)
                    self.name = "Spatio-Temporal Momentum Strategy"
                    self.description = "Strategy combining time-series and cross-sectional momentum"

                def generate_signals(self, data):
                    \"\"\"Generate trading signals based on momentum\"\"\"
                    df = data.copy()

                    # Ensure we have enough data
                    required_periods = self.momentum_lookback + self.momentum_lag + 1
                    if len(df) < required_periods:
                        df['signal'] = 0
                        return df

                    # Calculate time-series momentum
                    df['momentum'] = (
                        df['close'].shift(self.momentum_lag) / 
                        df['close'].shift(self.momentum_lookback + self.momentum_lag) - 1
                    )

                    # Generate signals based on momentum
                    df['signal'] = 0

                    # Buy signal: positive momentum above median
                    momentum_median = df['momentum'].rolling(window=60, min_periods=30).median()
                    buy_condition = (df['momentum'] > momentum_median) & (df['momentum'] > 0)
                    df.loc[buy_condition, 'signal'] = 1

                    # Sell signal: negative momentum below median
                    sell_condition = (df['momentum'] < momentum_median) & (df['momentum'] < 0)
                    df.loc[sell_condition, 'signal'] = -1

                    # Fill any NaN values in signal column
                    df['signal'] = df['signal'].fillna(0)

                    return df
            ```
            """
        else:
            # Generic mock response
            return """
            This is a mock response generated because no real LLM client is available.
            
            The system would normally query Claude API here, but either:
            1. The system is in dry run mode, or
            2. No valid API key was found, or
            3. The Claude client failed to initialize
            
            In a real run with proper API configuration, this would contain the actual response from Claude.
            """

    def query_claude(self, prompt, system_prompt=None, max_tokens=4000):
        """
        Query the Claude API with cost estimation and dry run support
        """
        # DEBUG INFO
        self.logger.info(f"DEBUG: Claude client available: {'claude' in self.clients}")
        self.logger.info(f"DEBUG: Dry run mode: {self.dry_run}")

        # Count tokens to estimate cost
        prompt_tokens = self._count_tokens(prompt)
        system_tokens = self._count_tokens(system_prompt) if system_prompt else 0
        total_input_tokens = prompt_tokens + system_tokens

        # Estimate cost (assumes max_tokens will be used)
        estimated_cost = self._estimate_cost(total_input_tokens, max_tokens, self.claude_model)

        self.logger.info(f"Estimated token usage - Input: {total_input_tokens}, Max output: {max_tokens}")
        self.logger.info(f"Estimated cost for this request: ${estimated_cost:.4f}")

        # Warn if this is an expensive call
        if estimated_cost > 0.10:
            self.logger.warning(f"⚠️ Expensive API call detected (${estimated_cost:.4f})")

        # Dry run mode - return a mock response
        if self.dry_run:
            self.logger.info("DRY RUN: Simulating Claude API call")
            prompt_type = "general"
            if "trading strategy" in prompt.lower() and "academic paper" in prompt.lower():
                prompt_type = "strategy_extraction"
            elif "trading strategy" in prompt.lower() and "implementation" in prompt.lower():
                prompt_type = "implementation"

            mock_response = self._get_mock_response(prompt_type)
            output_tokens = self._count_tokens(mock_response)
            self._log_token_usage(total_input_tokens, output_tokens, self.claude_model)
            return mock_response

        # Check if Claude client is available
        if "claude" not in self.clients:
            self.logger.error("Claude client not available - attempting to reinitialize")

            # Try to reinitialize the client
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                try:
                    import anthropic
                    self.clients["claude"] = anthropic.Anthropic(api_key=api_key)
                    self.logger.info("Successfully reinitialized Claude client")
                except Exception as e:
                    self.logger.error(f"Failed to reinitialize Claude client: {e}")
                    self.logger.warning("No LLM clients available, returning mock response")
                    mock_response = self._get_mock_response("general")
                    output_tokens = self._count_tokens(mock_response)
                    self._log_token_usage(total_input_tokens, output_tokens, self.claude_model)
                    return mock_response
            else:
                self.logger.error("No ANTHROPIC_API_KEY environment variable found")
                self.logger.warning("No LLM clients available, returning mock response")
                mock_response = self._get_mock_response("general")
                output_tokens = self._count_tokens(mock_response)
                self._log_token_usage(total_input_tokens, output_tokens, self.claude_model)
                return mock_response

        # Apply rate limiting
        self._apply_rate_limiting("claude")

        try:
            self.logger.info("Querying claude")

            messages = [{"role": "user", "content": prompt}]

            response = self.clients["claude"].messages.create(
                model=self.claude_model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages
            )

            response_text = response.content[0].text

            # Log actual token usage
            input_usage = response.usage.input_tokens
            output_usage = response.usage.output_tokens
            self._log_token_usage(input_usage, output_usage, self.claude_model)

            return response_text

        except Exception as e:
            self.logger.error(f"Claude API error: {e}")
            raise

    def query_local_model(self, provider, prompt, system_prompt=None, max_tokens=4000):
        """Query a local model like Llama3"""
        # Dry run mode
        if self.dry_run:
            self.logger.info(f"DRY RUN: Simulating {provider} API call")
            mock_response = self._get_mock_response()
            return mock_response

        # For real API calls, check if client is available
        if provider not in self.clients:
            raise ValueError(f"{provider} client not available")

        self._apply_rate_limiting(provider)

        try:
            # Implementation depends on the specific library used
            # Example for llama-cpp-python
            if provider == "llama3":
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                response = self.clients[provider].create_completion(
                    prompt=full_prompt,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    stop=["User:", "\n\n"]
                )
                return response["choices"][0]["text"]
        except Exception as e:
            self.logger.error(f"{provider} error: {e}")
            raise

    def query(self, prompt, system_prompt=None, max_tokens=4000, use_cache=True):
        """
        Query LLM models with fallback options
        """
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt + (system_prompt or ""))
            cached_response = self._check_cache(cache_key)
            if cached_response:
                self.logger.info("Using cached response")
                return cached_response

        # If no clients available and not in dry run, try to reinitialize
        if not self.clients and not self.dry_run:
            self.logger.warning("No LLM clients available, attempting to reinitialize...")

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                try:
                    import anthropic
                    self.clients["claude"] = anthropic.Anthropic(api_key=api_key)
                    self.logger.info("✅ Successfully reinitialized Claude client")
                except Exception as e:
                    self.logger.error(f"Failed to reinitialize Claude client: {e}")

        # Try primary provider
        primary_provider = self.primary_provider
        self.logger.info(f"Querying {primary_provider}")

        try:
            if primary_provider == "claude":
                response = self.query_claude(prompt, system_prompt, max_tokens)
            else:
                response = self.query_local_model(primary_provider, prompt, system_prompt, max_tokens)

            # Cache the response
            if use_cache:
                self._save_to_cache(cache_key, response)

            return response

        except Exception as e:
            self.logger.warning(f"Primary provider {primary_provider} failed: {e}")

            # Try fallback providers
            for provider in self.fallback_providers:
                if provider in self.clients or self.dry_run:
                    self.logger.info(f"Trying fallback provider {provider}")
                    try:
                        response = self.query_local_model(provider, prompt, system_prompt, max_tokens)
                        if use_cache:
                            self._save_to_cache(cache_key, response)
                        return response
                    except Exception as fallback_error:
                        self.logger.warning(f"Fallback provider {provider} failed: {fallback_error}")

            # If all providers fail, raise the original error
            self.logger.error(f"All LLM providers failed. Original error: {e}")
            raise e

    def get_strategy_extraction_prompt(self, title, abstract, content):
        """Generate a prompt for extracting a trading strategy from a paper"""
        return f"""
        You are a quantitative finance expert with extensive experience implementing academic trading strategies in Python. Extract the trading strategy described in the following academic paper.
        
        Paper Title: {title}
        Paper Abstract: {abstract}
        Paper Content: {content}
        
        Follow these steps:
        
        1. IDENTIFY the core trading signal or alpha generation mechanism
        2. LIST all specific indicators, factors, or features used with their definitions
        3. EXTRACT all mathematical formulas, ensuring they are correctly parsed
        4. SPECIFY all parameter values, lookback periods, and thresholds mentioned
        5. DETERMINE the asset classes, market conditions, and time frames where this strategy applies
        6. OUTLINE any risk management or position sizing rules
        
        Provide your output in the following JSON format:
        
        ```json
        {{
          "strategy_name": "Name of the strategy",
          "core_mechanism": "Detailed explanation of how the strategy generates alpha",
          "indicators": [
            {{
              "name": "Indicator name",
              "definition": "Mathematical or logical definition",
              "parameters": [List of parameters and their values]
            }}
          ],
          "key_formulas": [
            {{
              "description": "What this formula calculates",
              "latex": "LaTeX representation of the formula",
              "python_equivalent": "Python code implementing the formula"
            }}
          ],
          "asset_classes": ["List of applicable asset classes"],
          "market_conditions": ["List of optimal market conditions"],
          "time_frames": ["Applicable timeframes"],
          "risk_management": {{
            "position_sizing": "Method for determining position size",
            "stop_loss": "Stop loss rules if any",
            "risk_limits": "Any other risk constraints"
          }}
        }}
        ```
        
        Be precise and thorough. If you're unsure about any element, make your best estimate based on the information provided.
        """

    def get_implementation_prompt(self, strategy_overview):
        """Generate a prompt for implementing a trading strategy with standard format"""
        strategy_json = json.dumps(strategy_overview, indent=2)
        return f"""
        You are a quantitative developer experienced in implementing trading strategies in Python. Create a complete, executable Python implementation of the following trading strategy that STRICTLY follows our BaseStrategy format.

        Strategy Overview:
        {strategy_json}

        CRITICAL REQUIREMENTS - Your implementation MUST:

        1. Start with these EXACT imports:
        ```python
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'strategy_format'))
        from base_strategy import BaseStrategy
        import pandas as pd
        import numpy as np
        ```

        2. Create a class that inherits from BaseStrategy with this EXACT structure:
        ```python
        class [StrategyName](BaseStrategy):
            def _initialize_parameters(self, params):
                # Initialize ALL strategy parameters here
                pass

            def generate_signals(self, data):
                # Generate trading signals here
                df = data.copy()
                df['signal'] = 0  # Initialize signal column
                # Your signal logic here
                return df
        ```

        3. The class MUST implement these two abstract methods:
           - `_initialize_parameters(self, params)`: Initialize all strategy parameters
           - `generate_signals(self, data)`: Generate trading signals (1=buy, -1=sell, 0=hold)

        4. DO NOT implement backtest(), plot_results(), or _calculate_metrics() methods - these are provided by BaseStrategy

        5. Your generate_signals method MUST:
           - Accept a DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']
           - Return a DataFrame with all input columns plus a 'signal' column
           - Use 1 for buy signals, -1 for sell signals, 0 for hold

        6. Handle edge cases:
           - Check for sufficient data before calculating indicators
           - Use .fillna(0) for signal column to avoid NaN values
           - Ensure all calculations work with the provided data structure

        Now implement the complete strategy based on the provided overview. Include all relevant indicators and logic from the strategy overview, but follow the exact structure shown above.
        """

    def get_validation_prompt(self, implementation, issues):
        """Generate a prompt for fixing issues in a strategy implementation"""
        issues_text = "\n".join([f"- {issue}" for issue in issues])
        return f"""
        You are a quantitative developer experienced in implementing trading strategies in Python. Fix the following issues in this trading strategy implementation:
        
        Issues to address:
        {issues_text}
        
        Here is the current implementation:
        
        ```python
        {implementation}
        ```
        
        Provide a complete, corrected version of the code that addresses all the issues while maintaining the original strategy logic. Explain the changes you made to fix each issue.
        """

    def extract_strategy(self, paper, extract_content=None):
        """Extract a trading strategy from a research paper"""
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')

        # Get paper content if available or use extraction function
        content = ""
        if 'content' in paper and paper['content']:
            content = paper['content']
        elif extract_content:
            content = extract_content(paper)

        # If content is still empty, use a longer abstract
        if not content:
            content = abstract

        # Generate prompt for strategy extraction
        prompt = self.get_strategy_extraction_prompt(title, abstract, content)

        # Set system prompt for Claude
        system_prompt = "You are a quantitative finance expert specializing in extracting trading strategies from academic papers and converting them to practical implementations."

        # Query LLM
        self.logger.info(f"Extracting strategy from paper: {title}")
        response = self.query(prompt, system_prompt=system_prompt, max_tokens=4000)

        # Extract JSON from response
        try:
            # Find JSON block in the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without the markdown code block
                json_match = re.search(r'({[\s\S]*})', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    raise ValueError("No JSON found in response")

            # Parse JSON
            strategy_overview = json.loads(json_str)

            # Add metadata
            strategy_overview["paper_id"] = paper.get('id')
            strategy_overview["paper_title"] = title
            strategy_overview["paper_abstract"] = abstract
            strategy_overview["paper_link"] = paper.get('link')

            return strategy_overview

        except Exception as e:
            self.logger.error(f"Failed to extract strategy from response: {e}")
            self.logger.debug(f"Response: {response}")

            # Return partial info on failure
            return {
                "paper_id": paper.get('id'),
                "paper_title": title,
                "error": str(e),
                "raw_response": response
            }

    def generate_implementation(self, strategy_overview):
        """Generate Python code implementing a trading strategy"""
        # Generate prompt for implementation
        prompt = self.get_implementation_prompt(strategy_overview)

        # Set system prompt for Claude
        system_prompt = "You are a quantitative developer specializing in implementing trading strategies in Python. Your code should be complete, well-documented, and follow best practices."

        # Query LLM
        self.logger.info(f"Generating implementation for strategy: {strategy_overview.get('strategy_name', 'Unknown')}")
        response = self.query(prompt, system_prompt=system_prompt, max_tokens=8000)

        # Extract code from response
        try:
            # Find Python code block in the response
            code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                # Try to find the first code block of any type
                code_match = re.search(r'```(?:\w*)\s*(.*?)\s*```', response, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                else:
                    # If no code blocks found, use the entire response
                    code = response

            return code

        except Exception as e:
            self.logger.error(f"Failed to extract code from response: {e}")
            return f"# Error extracting code: {e}\n\n{response}"

    def refine_implementation(self, implementation, issues):
        """Refine a strategy implementation to fix issues"""
        # Generate prompt for validation
        prompt = self.get_validation_prompt(implementation, issues)

        # Set system prompt for Claude
        system_prompt = "You are a quantitative developer specializing in implementing trading strategies in Python. Your task is to fix issues in the provided code while maintaining the original strategy logic."

        # Query LLM
        self.logger.info(f"Refining implementation with {len(issues)} issues")
        response = self.query(prompt, system_prompt=system_prompt, max_tokens=8000)

        # Extract code from response
        try:
            # Find Python code block in the response
            code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                # Try to find the first code block of any type
                code_match = re.search(r'```(?:\w*)\s*(.*?)\s*```', response, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                else:
                    # If no code blocks found, use the entire response
                    code = response

            return code

        except Exception as e:
            self.logger.error(f"Failed to extract refined code from response: {e}")
            return f"# Error extracting refined code: {e}\n\n{response}"

    def get_token_usage_report(self):
        """Get a report of token usage and estimated costs"""
        return {
            "input_tokens": self.token_usage["input_tokens"],
            "output_tokens": self.token_usage["output_tokens"],
            "total_tokens": self.token_usage["input_tokens"] + self.token_usage["output_tokens"],
            "estimated_cost": self.token_usage["estimated_cost"],
            "model": self.claude_model
        }