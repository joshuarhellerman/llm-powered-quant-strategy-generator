#!/usr/bin/env python3
"""
Diagnostic script to find the exact line causing the segmentation fault
"""

import os
import sys
import importlib.util
import logging
import traceback
import signal
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def signal_handler(sig, frame):
    """Handler for segmentation fault signal"""
    logger.error("SEGMENTATION FAULT DETECTED!")
    logger.error(f"Last known position: {getattr(signal_handler, 'last_position', 'unknown')}")
    sys.exit(1)


# Register the signal handler for SIGSEGV
signal.signal(signal.SIGSEGV, signal_handler)


def test_tiktoken_directly():
    """Test tiktoken directly to see if it's the source of the issue"""
    logger.info("=== Testing tiktoken directly ===")

    try:
        logger.info("Checking if tiktoken is available...")
        if importlib.util.find_spec("tiktoken") is None:
            logger.info("tiktoken is not installed")
            return False

        logger.info("Importing tiktoken...")
        import tiktoken

        logger.info("Creating encoding...")
        enc = tiktoken.get_encoding("cl100k_base")

        logger.info("Encoding a test string...")
        test_str = "This is a test string for tiktoken"
        tokens = enc.encode(test_str)

        logger.info(f"Successfully encoded string with {len(tokens)} tokens")
        return True
    except Exception as e:
        logger.error(f"Error testing tiktoken: {e}")
        traceback.print_exc()
        return False


def diagnose_llm_service():
    """Test LLM service initialization step by step"""
    logger.info("=== Diagnosing LLMService initialization ===")

    try:
        # Add current directory to path if needed
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())

        logger.info("Step 1: Import LLMService class")
        signal_handler.last_position = "Importing LLMService"
        from trading.llm_service import LLMService

        logger.info("Step 2: Create minimal config")
        signal_handler.last_position = "Creating minimal config"
        config = {
            "dry_run": True,
            "primary_provider": "claude",
            "fallback_providers": [],
            "cache_dir": "output/llm_cache",
            "claude_model": "claude-3-5-haiku-20241022"
        }

        logger.info("Step 3: Initialize LLMService with config")
        signal_handler.last_position = "Initializing LLMService"
        service = LLMService(config)

        logger.info("Step 4: Test simple token counting")
        signal_handler.last_position = "Testing token counting"
        token_count = service._count_tokens("Test string")

        logger.info(f"Successfully counted tokens: {token_count}")
        return True
    except Exception as e:
        logger.error(f"Error diagnosing LLMService: {e}")
        traceback.print_exc()
        return False


def diagnose_run_pipeline():
    """Diagnose the run_pipeline function step by step"""
    logger.info("=== Diagnosing run_pipeline function ===")

    try:
        # Add current directory to path if needed
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())

        logger.info("Step 1: Import run_pipeline")
        signal_handler.last_position = "Importing run_pipeline"
        from main import run_pipeline, parse_arguments, setup_base_strategy, check_dependencies

        logger.info("Step 2: Test setup_base_strategy")
        signal_handler.last_position = "Testing setup_base_strategy"
        base_strategy_dir = setup_base_strategy()
        logger.info(f"BaseStrategy dir: {base_strategy_dir}")

        logger.info("Step 3: Test check_dependencies")
        signal_handler.last_position = "Testing check_dependencies"
        dependencies_ok = check_dependencies()
        logger.info(f"Dependencies check: {dependencies_ok}")

        logger.info("Step 4: Creating minimal args")
        signal_handler.last_position = "Creating minimal args"
        import argparse
        args = argparse.Namespace(
            config='config.yaml',
            mode='scrape',
            test=True,
            dry_run=True,
            limit=1,
            no_test=False,
            model=None,
            ensure_success=False,
            max_attempts=None
        )

        logger.info("Step 5: Running run_pipeline with minimal args")
        signal_handler.last_position = "Running run_pipeline"

        # This step will be executed line by line with prints
        logger.info("Step 5.1: Starting memory management")
        signal_handler.last_position = "Memory management"
        import gc
        gc.collect()

        logger.info("Step 5.2: Testing ConfigManager initialization")
        signal_handler.last_position = "Initializing ConfigManager"
        from trading.config_manager import ConfigManager
        config_manager = ConfigManager(args.config)
        logger.info("ConfigManager initialized")

        logger.info("Step 5.3: Getting output directories")
        signal_handler.last_position = "Getting output dirs"
        output_dirs = config_manager.get_output_dirs()
        logger.info(f"Output dirs: {output_dirs}")

        logger.info("Step 5.4: Getting scraping config")
        signal_handler.last_position = "Getting scraping config"
        scraping_config = config_manager.get_scraping_config()
        logger.info("Got scraping config")

        logger.info("Step 5.5: Getting test config")
        signal_handler.last_position = "Getting test config"
        test_config = config_manager.get_test_config()
        logger.info("Got test config")

        logger.info("Step 5.6: Initialize LLM service")
        signal_handler.last_position = "Initializing LLM service"
        from trading.llm_service import LLMService
        llm_config = config_manager.config.get('llm', {})
        if args.dry_run:
            llm_config['dry_run'] = True
        if args.model:
            model_mapping = {
                'haiku': 'claude-3-5-haiku-20241022',
                'sonnet': 'claude-3-7-sonnet-20250219',
                'opus': 'claude-3-opus-20240229'
            }
            if args.model in model_mapping:
                llm_config['claude_model'] = model_mapping[args.model]

        logger.info(f"LLM config: {llm_config}")
        llm_service = LLMService(llm_config)
        logger.info("LLM service initialized")

        logger.info("Step 5.7: Initialize validation service")
        signal_handler.last_position = "Initializing validation service"
        from trading.validation_service import ValidationService
        validation_service = ValidationService({
            "test_data_path": os.path.join(output_dirs['base_dir'], "test_data.csv")
        })
        logger.info("Validation service initialized")

        logger.info("Step 5.8: Initialize paper selection integrator")
        signal_handler.last_position = "Initializing paper selection integrator"
        from trading.paper_selection.integrator import PaperSelectionIntegrator
        paper_selection_config = config_manager.config.get('paper_selection', {})
        paper_selection_config['test_mode'] = True
        paper_selection_dir = os.path.join(output_dirs['papers_dir'], "selected")
        paper_selector_integrator = PaperSelectionIntegrator(
            llm_service=llm_service,
            config=paper_selection_config
        )
        logger.info("Paper selection integrator initialized")

        logger.info("Step 5.9: Initialize strategy extractor")
        signal_handler.last_position = "Initializing strategy extractor"
        from trading.strategy_extractor import StrategyExtractor
        strategies_dir = os.path.join(output_dirs['strategies_dir'], "llm_strategies")
        strategy_extractor = StrategyExtractor(
            llm_service=llm_service,
            validation_service=validation_service,
            config=config_manager.config,
            output_dir=strategies_dir,
            test_mode=True
        )
        logger.info("Strategy extractor initialized")

        logger.info("Step 5.10: Initialize strategy tester")
        signal_handler.last_position = "Initializing strategy tester"
        from trading.strategy_tester import StrategyTester
        strategy_tester = StrategyTester(
            strategies_dir=os.path.join(strategies_dir, "implementations"),
            test_data_path=os.path.join(output_dirs['base_dir'], "test_data.csv"),
            results_dir=os.path.join(output_dirs['base_dir'], "test_results")
        )
        logger.info("Strategy tester initialized")

        logger.info("All components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error during diagnosis: {e}")
        traceback.print_exc()
        return False


def diagnose_individual_modules():
    """Test each module independently to identify the problematic one"""
    logger.info("=== Diagnosing individual modules ===")

    modules = [
        "trading.config_manager",
        "trading.scraper",
        "trading.analyzer",
        "trading.llm_service",
        "trading.strategy_extractor",
        "trading.validation_service",
        "trading.strategy_tester",
        "trading.paper_selection.integrator"
    ]

    for module_name in modules:
        logger.info(f"Testing import of {module_name}...")
        signal_handler.last_position = f"Importing {module_name}"

        try:
            module = importlib.import_module(module_name)
            logger.info(f"Successfully imported {module_name}")

            # Try to get the main class from the module
            if module_name == "trading.config_manager":
                cls = module.ConfigManager
                logger.info("Testing ConfigManager initialization...")
                instance = cls("config.yaml")
                logger.info("ConfigManager initialized successfully")
            elif module_name == "trading.llm_service":
                cls = module.LLMService
                logger.info("Testing LLMService initialization...")
                instance = cls({"dry_run": True})
                logger.info("LLMService initialized successfully")
            else:
                # Just report success for other modules
                logger.info(f"Import test for {module_name} succeeded")
        except Exception as e:
            logger.error(f"Error testing {module_name}: {e}")
            traceback.print_exc()
            logger.error(f"Module {module_name} might be the cause of the segfault")
            return module_name

    logger.info("All modules imported successfully")
    return None


def main():
    """Main diagnostic function"""
    logger.info("Starting detailed segmentation fault diagnosis")

    # Test tiktoken directly first
    tiktoken_ok = test_tiktoken_directly()

    if not tiktoken_ok:
        logger.warning("tiktoken test failed - this might be the source of the segfault")

    # First diagnose individual modules
    problematic_module = diagnose_individual_modules()

    if problematic_module:
        logger.warning(f"Found problematic module: {problematic_module}")

        # If the problematic module is llm_service, diagnose it in detail
        if problematic_module == "trading.llm_service":
            diagnose_llm_service()
    else:
        # If no single module is problematic, try the LLM service specifically
        llm_service_ok = diagnose_llm_service()

        if not llm_service_ok:
            logger.warning("LLM service initialization failed - likely the source of the segfault")
        else:
            # If LLM service is fine, try the run_pipeline function
            pipeline_ok = diagnose_run_pipeline()

            if not pipeline_ok:
                logger.warning("run_pipeline function failed - likely the source of the segfault")
            else:
                logger.info("All components tested successfully - couldn't identify the specific cause")

    logger.info("Diagnosis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())