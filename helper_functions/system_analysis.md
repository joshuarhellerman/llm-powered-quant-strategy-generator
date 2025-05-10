# System Analysis: helper_functions

## Overview

- **Project Directory**: .
- **Files**: 1
- **Directories**: 1

## Important Files

No important project files detected.


## Project Structure

```
helper_functions/
└── analyze_system.py
```

### Largest Directories

| Directory | Files | Size |
|-----------|-------|------|
| helper_functions | 1 | 18.1 KB |


## Dependencies

### Module Relationships


## Documentation

### `analyze_system.py`

Simple System Analyzer

This script analyzes a project directory and generates a clear report showing:
1. Project structure (directories and files)
2. Dependency relationships between modules
3. Basic documentation of the codebase

It outputs a single detailed markdown file with all the information.

#### Functions

##### `analyze_system`

Analyze the project and generate a comprehensive report.

Args:
    project_dir: Path to the project directory
    output_file: Path to the output file
    exclude_dirs: List of directories to exclude

##### `get_project_structure`

Get the complete directory structure with file sizes.

##### `extract_dependencies`

Extract module dependencies in the project.

##### `find_important_files`

Find important project files like README, configuration files, etc.

##### `extract_documentation`

Extract docstrings and documentation from Python files.

##### `format_size`

Format bytes to human-readable size.

##### `write_report`

Write the complete analysis report.

##### `write_directory_tree`

Write a directory tree in text format.


## Insights & Recommendations

- Add a README file to document the project.
