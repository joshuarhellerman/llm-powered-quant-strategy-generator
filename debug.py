#!/usr/bin/env python3
"""
Crash detector for main.py using faulthandler to identify
the location of segmentation faults
"""

import os
import sys
import faulthandler
import argparse
import traceback
import importlib
import logging

# Enable faulthandler to get traceback on crashes
faulthandler.enable()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_crash_log():
    """Setup crash log file for faulthandler"""
    crash_log = "crash_log.txt"
    logger.info(f"Setting up crash log at {crash_log}")
    crash_fd = open(crash_log, "w")
    faulthandler.enable(crash_fd)
    return crash_fd


def import_main_without_running():
    """Import main module without executing main() function"""
    try:
        # Add current directory to path if needed
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())

        # Try to import main module without running it
        spec = importlib.util.spec_from_file_location("main_module", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        logger.info("Successfully imported main.py module")
        return main_module
    except Exception as e:
        logger.error(f"Error importing main.py: {e}")
        traceback.print_exc()
        return None


def test_function(main_module, function_name, *args, **kwargs):
    """Test a specific function from the main module"""
    try:
        if hasattr(main_module, function_name):
            func = getattr(main_module, function_name)
            logger.info(f"Testing function: {function_name}")
            result = func(*args, **kwargs)
            logger.info(f"Function {function_name} completed successfully")
            return result
        else:
            logger.error(f"Function {function_name} not found in main module")
            return None
    except Exception as e:
        logger.error(f"Error executing {function_name}: {e}")
        traceback.print_exc()
        return None


def monitor_memory_usage():
    """Monitor memory usage during execution"""
    try:
        import psutil
        process = psutil.Process()
        logger.info(f"Initial memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
        return process
    except ImportError:
        logger.warning("psutil not available, memory monitoring disabled")
        return None


def run_with_args(args):
    """Run main.py with specific arguments using subprocess"""
    import subprocess
    import sys

    cmd = [sys.executable, "main.py"]
    for arg_name, arg_value in vars(args).items():
        if isinstance(arg_value, bool):
            if arg_value:
                cmd.append(f"--{arg_name}")
        else:
            if arg_value is not None:
                cmd.append(f"--{arg_name}={arg_value}")

    logger.info(f"Running command: {' '.join(cmd)}")

    # Run with environment variable to enable faulthandler traceback
    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )

        stdout, stderr = proc.communicate()

        logger.info(f"Process exited with code {proc.returncode}")

        if stdout:
            logger.info("STDOUT:")
            for line in stdout.splitlines():
                logger.info(f"  {line}")

        if stderr:
            logger.info("STDERR:")
            for line in stderr.splitlines():
                logger.info(f"  {line}")

        return proc.returncode == 0
    except Exception as e:
        logger.error(f"Error running subprocess: {e}")
        traceback.print_exc()
        return False


def main():
    """Main function"""
    logger.info("Starting crash detector for main.py")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Crash detector for main.py")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--function", help="Test a specific function from main.py")
    args = parser.parse_args()

    # Setup crash log
    crash_fd = setup_crash_log()

    # Monitor memory usage
    process = monitor_memory_usage()

    if args.interactive:
        logger.info("Running in interactive mode")

        # Import main module
        main_module = import_main_without_running()
        if main_module is None:
            logger.error("Failed to import main.py")
            return 1

        # Interactive testing of specific functions
        while True:
            print("\nAvailable functions:")
            functions = [name for name in dir(main_module) if
                         callable(getattr(main_module, name)) and not name.startswith("_")]
            for i, func in enumerate(functions):
                print(f"{i + 1}. {func}")
            print("0. Exit")

            try:
                choice = int(input("Enter function number to test (0 to exit): "))
                if choice == 0:
                    break

                if 1 <= choice <= len(functions):
                    func_name = functions[choice - 1]
                    test_function(main_module, func_name)

                    # Log memory usage if available
                    if process:
                        logger.info(
                            f"Memory usage after {func_name}: {process.memory_info().rss / (1024 * 1024):.2f} MB")
                else:
                    print("Invalid choice")
            except ValueError:
                print("Please enter a number")
            except Exception as e:
                logger.error(f"Error: {e}")
                traceback.print_exc()
    elif args.function:
        # Test a specific function
        main_module = import_main_without_running()
        if main_module is None:
            logger.error("Failed to import main.py")
            return 1

        test_function(main_module, args.function)

        # Log memory usage if available
        if process:
            logger.info(f"Memory usage after {args.function}: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    else:
        # Run with minimal test arguments
        test_args = argparse.Namespace(
            config="config.yaml",
            mode="scrape",
            test=True,
            dry_run=True,
            limit=1,
            no_test=False,
            model=None,
            ensure_success=False,
            max_attempts=None
        )

        success = run_with_args(test_args)
        if not success:
            logger.error("Failed to run main.py with minimal test arguments")
            return 1

    # Close crash log
    crash_fd.close()

    logger.info("Crash detector completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())