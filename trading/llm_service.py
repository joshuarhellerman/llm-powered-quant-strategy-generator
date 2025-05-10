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
import tiktoken  # For token counting
from typing import Dict, List, Any, Optional, Union
import anthropic
import importlib.util


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
        self.claude_model = self.config.get("claude_model", "claude-3-7-sonnet-20250219")
        self.logger.info(f"Using Claude model: {self.claude_model}")

        # Token and cost estimation parameters
        self.model_costs = {
            "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},   # $1.00/$5.00 per million tokens
            "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},    # $15.00/$75.00 per million tokens
            "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00}  # $3.00/$15.00 per million tokens
        }

        # Setup API keys from config or environment variables
        self.api_keys = self.config.get("api_keys", {})
        if not self.api_keys.get("claude") and os.environ.get("ANTHROPIC_API_KEY"):
            self.api_keys["claude"] = os.environ.get("ANTHROPIC_API_KEY")

        # Setup clients
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

        # Set up tokenizer for counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Claude uses cl100k_base
            self.logger.info("Initialized tiktoken for token counting")
        except:
            self.logger.warning("Failed to initialize tiktoken. Token counts will be estimated.")
            self.tokenizer = None

    def _setup_clients(self):
        """Setup LLM clients based on available providers"""
        self.clients = {}

        # Skip client setup if in dry run mode
        if self.dry_run:
            self.logger.info("Dry run mode: Skipping actual client initialization")
            return

        # Setup Claude client if API key is available
        if "claude" in self.api_keys:
            try:
                self.clients["claude"] = anthropic.Anthropic(api_key=self.api_keys["claude"])
                self.logger.info("Claude client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Claude client: {e}")

        # Setup local models if specified
        for provider in self.fallback_providers:
            if provider == "llama3" and self.config.get("local_models", {}).get("llama3_path"):
                # This would import and setup a local Llama model
                # Implementation would depend on how you're hosting Llama
                try:
                    self.logger.info(f"Setting up local {provider} model")
                    # Example if using llama-cpp-python
                    if importlib.util.find_spec("llama_cpp"):
                        import llama_cpp
                        model_path = self.config["local_models"]["llama3_path"]
                        self.clients[provider] = llama_cpp.Llama(model_path=model_path)
                        self.logger.info(f"Local {provider} model initialized successfully")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {provider} client: {e}")

    def _apply_rate_limiting(self, provider):
        """Apply rate limiting for the specified provider"""
        if provider not in self.rate_limits:
            return

        rate_limit = self.rate_limits[provider]
        current_time = time.time()
        time_since_last_request = current_time - rate_limit["last_request_time"]

        # Calculate minimum time between requests
        min_interval = 60.0 / rate_limit["requests_per_minute"]

        # Sleep if needed
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            self.logger.debug(f"Rate limiting {provider}: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        # Update last request time
        self.rate_limits[provider]["last_request_time"] = time.time()

    def _count_tokens(self, text):
        """
        Count the number of tokens in a text string

        Args:
            text (str): Text to count tokens for

        Returns:
            int: Number of tokens
        """
        if text is None:
            return 0

        if self.tokenizer:
            # Use tiktoken for accurate token counting
            return len(self.tokenizer.encode(text))
        else:
            # Fallback to a simple approximation (Claude uses about 4 chars per token on average)
            return len(text) // 4

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
            DRY RUN MODE - This is a mock response for strategy extraction.
            
            ```json
            {
              "strategy_name": "Mock Momentum Strategy",
              "core_mechanism": "This momentum strategy uses price and volume trends to identify trading opportunities.",
              "indicators": [
                {
                  "name": "EMA Crossover",
                  "definition": "Crossover of fast and slow exponential moving averages",
                  "parameters": [10, 50]
                },
                {
                  "name": "RSI",
                  "definition": "Relative Strength Index for overbought/oversold conditions",
                  "parameters": [14]
                }
              ],
              "key_formulas": [
                {
                  "description": "Momentum calculation",
                  "latex": "M_t = \\frac{P_t}{P_{t-n}}",
                  "python_equivalent": "momentum = price / price.shift(n)"
                }
              ],
              "asset_classes": ["Stocks", "ETFs", "Futures"],
              "market_conditions": ["Trending markets", "High volatility"],
              "time_frames": ["Daily", "Weekly"],
              "risk_management": {
                "position_sizing": "2% of portfolio per trade",
                "stop_loss": "5% below entry price",
                "risk_limits": "Maximum 20% portfolio exposure"
              }
            }
            ```
            """
        elif prompt_type == "implementation":
            return """
            DRY RUN MODE - This is a mock response for strategy implementation.
            
            ```python
            from strategy_format.base_strategy import BaseStrategy
            import pandas as pd
            import numpy as np
            
            class MockMomentumStrategy(BaseStrategy):
                \"\"\"
                Mock Momentum Strategy for testing purposes.
                
                Based on: Mock Research Paper
                Authors: Mock Author
                \"\"\"
                
                def _initialize_parameters(self, params):
                    \"\"\"Initialize strategy parameters\"\"\"
                    self.fast_window = params.get('fast_window', 10)
                    self.slow_window = params.get('slow_window', 50)
                    self.rsi_window = params.get('rsi_window', 14)
                    self.rsi_oversold = params.get('rsi_oversold', 30)
                    self.rsi_overbought = params.get('rsi_overbought', 70)
                
                def generate_signals(self, data):
                    \"\"\"Generate trading signals\"\"\"
                    # Make a copy of the input data
                    df = data.copy()
                    
                    # Calculate indicators
                    df['ema_fast'] = df['close'].ewm(span=self.fast_window, adjust=False).mean()
                    df['ema_slow'] = df['close'].ewm(span=self.slow_window, adjust=False).mean()
                    df['rsi'] = self._calculate_rsi(df['close'], self.rsi_window)
                    
                    # Generate signals
                    df['signal'] = 0  # Default to hold
                    
                    # Buy signal: Fast EMA crosses above Slow EMA and RSI < overbought
                    buy_condition = (
                        (df['ema_fast'] > df['ema_slow']) & 
                        (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) &
                        (df['rsi'] < self.rsi_overbought)
                    )
                    df.loc[buy_condition, 'signal'] = 1
                    
                    # Sell signal: Fast EMA crosses below Slow EMA or RSI > overbought
                    sell_condition = (
                        (df['ema_fast'] < df['ema_slow']) & 
                        (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)) |
                        (df['rsi'] > self.rsi_overbought)
                    )
                    df.loc[sell_condition, 'signal'] = -1
                    
                    return df
                
                def _calculate_rsi(self, prices, window):
                    \"\"\"Calculate RSI indicator\"\"\"
                    # Calculate price changes
                    delta = prices.diff()
                    
                    # Separate gains and losses
                    gains = delta.copy()
                    losses = delta.copy()
                    gains[gains < 0] = 0
                    losses[losses > 0] = 0
                    losses = -losses
                    
                    # Calculate averages
                    avg_gain = gains.rolling(window=window).mean()
                    avg_loss = losses.rolling(window=window).mean()
                    
                    # Calculate RS and RSI
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    return rsi
            ```
            """
        else:
            # Generic mock response
            return """
            DRY RUN MODE - This is a mock response.
            
            The system is operating in dry run mode, so no actual API calls are being made.
            This response is a placeholder to allow testing the pipeline without using API credits.
            
            In a real run, this would contain the actual response from the Claude API.
            """

    def query_claude(self, prompt, system_prompt=None, max_tokens=4000):
        """
        Query the Claude API with cost estimation and dry run support

        Args:
            prompt (str): User prompt
            system_prompt (str): Optional system prompt
            max_tokens (int): Maximum tokens to generate

        Returns:
            str: Claude's response
        """
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

            # Determine what type of prompt this is to generate an appropriate mock response
            prompt_type = "general"
            if "trading strategy" in prompt.lower() and "academic paper" in prompt.lower():
                prompt_type = "strategy_extraction"
            elif "trading strategy" in prompt.lower() and "implementation" in prompt.lower():
                prompt_type = "implementation"

            mock_response = self._get_mock_response(prompt_type)

            # Log simulated token usage
            output_tokens = self._count_tokens(mock_response)
            self._log_token_usage(total_input_tokens, output_tokens, self.claude_model)

            return mock_response

        # For real API calls, check if client is available
        if "claude" not in self.clients:
            raise ValueError("Claude client not available")

        # Apply rate limiting
        self._apply_rate_limiting("claude")

        try:
            messages = [{"role": "user", "content": prompt}]

            response = self.clients["claude"].messages.create(
                model=self.claude_model,  # Use the configured model
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

        Args:
            prompt (str): The prompt to send to the LLM
            system_prompt (str): Optional system prompt
            max_tokens (int): Maximum number of tokens in the response
            use_cache (bool): Whether to use cache for the query

        Returns:
            str: The model's response
        """
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt + (system_prompt or ""))
            cached_response = self._check_cache(cache_key)
            if cached_response:
                self.logger.info("Using cached response")
                return cached_response

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
                if provider in self.clients or self.dry_run:  # Allow fallback in dry run mode
                    self.logger.info(f"Trying fallback provider {provider}")
                    try:
                        response = self.query_local_model(provider, prompt, system_prompt, max_tokens)

                        # Cache the response
                        if use_cache:
                            self._save_to_cache(cache_key, response)

                        return response
                    except Exception as fallback_error:
                        self.logger.warning(f"Fallback provider {provider} failed: {fallback_error}")

            # If all providers fail, raise the original error
            raise e

    def get_strategy_extraction_prompt(self, title, abstract, content):
        """
        Generate a prompt for extracting a trading strategy from a paper

        Args:
            title (str): Paper title
            abstract (str): Paper abstract
            content (str): Paper content

        Returns:
            str: Prompt for the LLM
        """
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
        """
        Generate a prompt for implementing a trading strategy with standard format

        Args:
            strategy_overview (dict): Strategy overview extracted from the paper

        Returns:
            str: Prompt for the LLM
        """
        # Convert the strategy overview to a JSON string
        strategy_json = json.dumps(strategy_overview, indent=2)

        return f"""
        You are a quantitative developer experienced in implementing trading strategies in Python. Create a complete, executable Python implementation of the following trading strategy that follows our standard format specification.

        Strategy Overview:
        {strategy_json}

        Your implementation MUST follow these format requirements:
        
        1. The strategy class MUST inherit from the BaseStrategy class
        2. The strategy class MUST implement these required methods:
           - _initialize_parameters(self, params): Initialize strategy parameters
           - generate_signals(self, data): Generate trading signals (1=buy, -1=sell, 0=hold)
        
        3. The class must follow this structure:
        
        ```python
        from strategy_format.base_strategy import BaseStrategy
        import pandas as pd
        import numpy as np
        
        class StrategyName(BaseStrategy):
            \"\"\"
            Strategy description based on the research paper.
            
            Based on: [Paper Title]
            Authors: [Paper Authors]
            \"\"\"
            
            def _initialize_parameters(self, params):
                \"\"\"Initialize strategy parameters\"\"\"
                # Initialize all strategy-specific parameters here
                self.parameter1 = params.get('parameter1', default_value)
                self.parameter2 = params.get('parameter2', default_value)
                # ...
            
            def generate_signals(self, data):
                \"\"\"Generate trading signals\"\"\"
                # Make a copy of the input data
                df = data.copy()
                
                # IMPLEMENT THE STRATEGY LOGIC HERE
                # Calculate indicators, signals, etc.
                # ...
                
                # Generate signals (1=buy, -1=sell, 0=hold)
                df['signal'] = 0  # Default to hold
                
                # Example signal logic - replace with actual strategy
                # df.loc[condition_for_buy, 'signal'] = 1
                # df.loc[condition_for_sell, 'signal'] = -1
                
                return df
        ```
        
        All your calculations must happen within the generate_signals method. The BaseStrategy will handle backtesting, metrics calculation, and visualization.

        Do not implement the backtest method, as it's already provided by the BaseStrategy class.
        
        Write a complete Python class that implements this strategy correctly, including:
        - Proper imports
        - Full implementation of all indicators mentioned in the strategy
        - Precise implementation of the key mathematical formulas
        - Proper signal generation based on the strategy logic
        
        The code should:
        - Handle edge cases properly (e.g., NaN values, lookback periods)
        - Include clear documentation for all methods
        - Follow PEP 8 standards for Python code
        
        Do not abbreviate any implementation. Provide the full, executable code.
        """

    def get_validation_prompt(self, implementation, issues):
        """
        Generate a prompt for fixing issues in a strategy implementation

        Args:
            implementation (str): The strategy implementation
            issues (list): List of issues to fix

        Returns:
            str: Prompt for the LLM
        """
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
        """
        Extract a trading strategy from a research paper

        Args:
            paper (dict): Paper dictionary with metadata
            extract_content (function): Optional function to extract content from paper

        Returns:
            dict: Extracted strategy information
        """
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
        """
        Generate Python code implementing a trading strategy

        Args:
            strategy_overview (dict): Strategy overview extracted from the paper

        Returns:
            str: Python code implementing the strategy
        """
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
        """
        Refine a strategy implementation to fix issues

        Args:
            implementation (str): The strategy implementation
            issues (list): List of issues to fix

        Returns:
            str: Refined implementation
        """
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
        """
        Get a report of token usage and estimated costs

        Returns:
            dict: Token usage statistics
        """
        return {
            "input_tokens": self.token_usage["input_tokens"],
            "output_tokens": self.token_usage["output_tokens"],
            "total_tokens": self.token_usage["input_tokens"] + self.token_usage["output_tokens"],
            "estimated_cost": self.token_usage["estimated_cost"],
            "model": self.claude_model
        }