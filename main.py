"""
Main script for the Trading Strategy Paper Scraper with LLM integration
Refactored to use PipelineStages for better organization
"""

import os
import argparse
import logging
import sys
import time
import traceback

# Set up fault handler for segmentation faults
try:
    import faulthandler
    faulthandler.enable()
    print("Faulthandler enabled for crash diagnostics")
except ImportError:
    print("Warning: faulthandler module not available - no crash diagnostics will be generated")


def setup_logging(log_level, log_file=None):
    """
    Setup logging configuration

    Args:
        log_level (str): Log level (INFO, DEBUG, etc.)
        log_file (str): Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    level = getattr(logging, log_level.upper(), logging.INFO)

    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(level=level, format=log_format,
                            handlers=[
                                logging.FileHandler(log_file),
                                logging.StreamHandler()
                            ])
    else:
        logging.basicConfig(level=level, format=log_format)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Trading Strategy Paper Scraper with LLM')

    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', default='all', choices=['scrape', 'analyze', 'select', 'extract', 'test', 'all'],
                        help='Operation mode')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with limited papers (overrides config.yaml)')
    parser.add_argument('--no-test', action='store_true', help='Force disable test mode (overrides config.yaml)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of papers to process')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry run mode without making actual API calls')
    parser.add_argument('--model', choices=['haiku', 'sonnet', 'opus'], default=None,
                        help='Claude model to use (haiku is cheapest, opus is most expensive)')
    parser.add_argument('--ensure-success', action='store_true',
                        help='In test mode, keep trying until finding a paper with implementable strategy')
    parser.add_argument('--max-attempts', type=int, default=None,
                        help='Maximum number of attempts when using --ensure-success')
    parser.add_argument('--debug', action='store_true', help='Enable extra debug output')

    return parser.parse_args()


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    required = ['pandas', 'numpy', 'matplotlib', 'requests', 'bs4', 'nltk', 'tqdm', 'yaml', 'anthropic']

    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package}")

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install them using pip:")
        print(f"pip install {' '.join(missing)}")
        return False

    # Check for tiktoken (used for token counting)
    try:
        __import__('tiktoken')
        print("✓ tiktoken")
    except ImportError:
        print("Warning: tiktoken is not installed. Token counting will use estimations.")
        print("To install: pip install tiktoken")

    print("All dependencies checked")
    return True


def import_pipeline_modules():
    """Import pipeline modules with error handling"""
    try:
        print("Importing config_manager...")
        from trading.config_manager import ConfigManager

        print("Importing pipeline_stages...")
        from pipeline_stages import PipelineStages

        print("All modules imported successfully")
        return ConfigManager, PipelineStages
    except Exception as e:
        print(f"ERROR during module import: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point"""
    print("=== Starting Trading Strategy Paper Scraper ===")

    # Record start time for performance tracking
    start_time = time.time()

    try:
        # Parse arguments
        print("Parsing command line arguments...")
        args = parse_arguments()
        print(f"Arguments: {args}")

        # Setup debugging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            print("DEBUG mode enabled - using verbose logging")

        # Check dependencies
        if not check_dependencies():
            return 1

        # Import required modules
        ConfigManager, PipelineStages = import_pipeline_modules()

        # Load configuration
        print("Loading configuration...")
        try:
            config_manager = ConfigManager(args.config)
            output_dirs = config_manager.get_output_dirs()
            print(f"Config loaded, output dirs: {output_dirs}")
        except Exception as e:
            print(f"ERROR loading configuration: {e}")
            traceback.print_exc()
            return 1

        # Setup logging
        print("Setting up logging...")
        log_level = config_manager.get_output_config().get('log_level', 'INFO')
        log_file = os.path.join(output_dirs['base_dir'], 'pipeline.log')
        setup_logging(log_level, log_file)

        # Initialize pipeline stages
        print("Initializing pipeline stages...")
        try:
            pipeline = PipelineStages(config_manager, output_dirs)
        except Exception as e:
            print(f"ERROR initializing pipeline stages: {e}")
            traceback.print_exc()
            return 1

        # Run the pipeline
        print("Starting pipeline execution...")
        try:
            results = pipeline.run_full_pipeline(args)

            if results is None:
                print("Pipeline returned None - execution failed")
                return 1

            print("✅ Pipeline completed successfully")

            # Calculate and display execution time
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Total execution time: {execution_time:.2f} seconds ({execution_time / 60:.2f} minutes)")

            return 0

        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}", exc_info=True)
            print(f"PIPELINE ERROR: {e}")
            traceback.print_exc()
            return 1

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    print(f"Exiting with code {exit_code}")
    sys.exit(exit_code)