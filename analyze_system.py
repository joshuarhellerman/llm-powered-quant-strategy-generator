#!/usr/bin/env python3
"""
Simple System Analyzer

This script analyzes a project directory and generates a clear report showing:
1. Project structure (directories and files)
2. Dependency relationships between modules
3. Basic documentation of the codebase

It outputs a single detailed markdown file with all the information.
"""

import os
import re
import sys
import ast
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter


def analyze_system(project_dir, output_file, exclude_dirs=None):
    """
    Analyze the project and generate a comprehensive report.

    Args:
        project_dir: Path to the project directory
        output_file: Path to the output file
        exclude_dirs: List of directories to exclude
    """
    if exclude_dirs is None:
        exclude_dirs = ['venv', 'env', '.git', '__pycache__', 'node_modules', '.idea', '.vscode', 'build', 'dist']

    # Collect data
    project_structure = get_project_structure(project_dir, exclude_dirs)
    dependencies = extract_dependencies(project_dir, exclude_dirs)
    important_files = find_important_files(project_dir)
    documentation = extract_documentation(project_dir, exclude_dirs)

    # Generate the report
    with open(output_file, 'w') as f:
        write_report(f, project_dir, project_structure, dependencies, important_files, documentation)

    print(f"Analysis complete. Report saved to: {output_file}")


def get_project_structure(project_dir, exclude_dirs):
    """Get the complete directory structure with file sizes."""
    structure = {}
    file_count = 0
    dir_count = 0

    for root, dirs, files in os.walk(project_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        rel_path = os.path.relpath(root, project_dir)
        if rel_path == '.':
            rel_path = ''

        structure[rel_path] = {
            'type': 'directory',
            'files': [],
            'size': 0,
            'file_count': 0
        }

        dir_count += 1

        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)

                structure[rel_path]['files'].append({
                    'name': file,
                    'size': size,
                    'extension': os.path.splitext(file)[1].lower()
                })

                structure[rel_path]['size'] += size
                structure[rel_path]['file_count'] += 1
                file_count += 1
            except:
                continue

    return {
        'structure': structure,
        'file_count': file_count,
        'dir_count': dir_count
    }


def extract_dependencies(project_dir, exclude_dirs):
    """Extract module dependencies in the project."""
    dependencies = defaultdict(set)
    reverse_dependencies = defaultdict(set)
    python_files = []

    # Find all Python files
    for root, dirs, files in os.walk(project_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_dir)
                python_files.append((rel_path, file_path))

    # Process each Python file
    for rel_path, abs_path in python_files:
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Parse the AST
            try:
                tree = ast.parse(content)

                # Find all imports
                imports = []
                from_imports = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:  # Ignore relative imports with no module
                            from_imports.append(node.module)

                # Check for project modules (not standard lib)
                for module in imports + from_imports:
                    parts = module.split('.')

                    # Look for modules that correspond to project files
                    for i in range(len(parts), 0, -1):
                        prefix = '.'.join(parts[:i])
                        for other_rel_path, _ in python_files:
                            other_module = os.path.splitext(other_rel_path)[0].replace('/', '.').replace('\\', '.')
                            if other_module == prefix or other_module.endswith('.' + prefix):
                                dependencies[rel_path].add(other_rel_path)
                                reverse_dependencies[other_rel_path].add(rel_path)
                                break
            except SyntaxError:
                # Fall back to regex for files with syntax errors
                import_pattern = r'^(?:from\s+(\S+)\s+import|import\s+(\S+))'
                for line in content.splitlines():
                    m = re.match(import_pattern, line)
                    if m:
                        module = m.group(1) or m.group(2)
                        # Basic handling - just store the import without resolution
                        dependencies[rel_path].add(module)
        except:
            pass

    # Find modules with circular dependencies
    circular = []
    visited = set()

    def find_circular(node, path=None):
        if path is None:
            path = []

        if node in path:
            # Found a cycle
            cycle_start = path.index(node)
            circular.append(path[cycle_start:] + [node])
            return

        path.append(node)
        for dep in dependencies.get(node, []):
            if isinstance(dep, str) and os.path.isfile(os.path.join(project_dir, dep)):
                find_circular(dep, path.copy())

    for node in dependencies:
        if node not in visited:
            find_circular(node)
            visited.add(node)

    # Identify critical modules (most depended upon)
    critical = []
    for module, deps in reverse_dependencies.items():
        if len(deps) >= 3:  # At least 3 other modules depend on this
            critical.append({
                'module': module,
                'dependents': len(deps)
            })

    critical.sort(key=lambda x: x['dependents'], reverse=True)

    return {
        'dependencies': {k: list(v) for k, v in dependencies.items()},
        'reverse_dependencies': {k: list(v) for k, v in reverse_dependencies.items()},
        'circular': circular[:10],  # Limit to 10 for readability
        'critical': critical[:10]  # Limit to 10 for readability
    }


def find_important_files(project_dir):
    """Find important project files like README, configuration files, etc."""
    important_patterns = {
        'readme': [r'^readme\.md$', r'^readme\.txt$', r'^readme$', r'^README\.md$'],
        'license': [r'^license(\.\w+)?$', r'^LICENSE(\.\w+)?$'],
        'requirements': [r'^requirements\.txt$', r'^pyproject\.toml$', r'^Pipfile$'],
        'package.json': [r'^package\.json$'],
        'setup.py': [r'^setup\.py$'],
        'dockerfile': [r'^Dockerfile$', r'^docker-compose\.ya?ml$'],
        'configuration': [r'^\.env(\.\w+)?$', r'.*\.config$', r'.*\.cfg$', r'.*\.conf$'],
        'ci': [r'^\.github/workflows/.*\.ya?ml$', r'^\.gitlab-ci\.ya?ml$', r'^\.travis\.yml$'],
        'entry_point': [r'^main\.py$', r'^app\.py$', r'^run\.py$', r'^manage\.py$', r'^index\.(js|ts|php)$']
    }

    important = {}

    for root, _, files in os.walk(project_dir):
        for file in files:
            # Check if file matches any patterns
            for file_type, patterns in important_patterns.items():
                if any(re.match(pattern, file, re.IGNORECASE) for pattern in patterns):
                    rel_path = os.path.relpath(os.path.join(root, file), project_dir)
                    if file_type not in important:
                        important[file_type] = []
                    important[file_type].append(rel_path)

    return important


def extract_documentation(project_dir, exclude_dirs):
    """Extract docstrings and documentation from Python files."""
    docs = {}

    for root, dirs, files in os.walk(project_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_dir)

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    try:
                        # Parse the AST
                        tree = ast.parse(content)

                        # Extract module docstring
                        module_doc = ast.get_docstring(tree)

                        # Extract classes and functions with docstrings
                        classes = {}
                        functions = {}

                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                class_doc = ast.get_docstring(node)
                                if class_doc:
                                    classes[node.name] = class_doc
                            elif isinstance(node, ast.FunctionDef):
                                func_doc = ast.get_docstring(node)
                                if func_doc:
                                    functions[node.name] = func_doc

                        # Only store files that have some documentation
                        if module_doc or classes or functions:
                            docs[rel_path] = {
                                'module_doc': module_doc,
                                'classes': classes,
                                'functions': functions
                            }
                    except SyntaxError:
                        pass
                except:
                    pass

    return docs


def format_size(size_bytes):
    """Format bytes to human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def write_report(f, project_dir, structure, dependencies, important_files, documentation):
    """Write the complete analysis report."""
    project_name = os.path.basename(os.path.abspath(project_dir))

    # Write header
    f.write(f"# System Analysis: {project_name}\n\n")
    f.write("## Overview\n\n")
    f.write(f"- **Project Directory**: {project_dir}\n")
    f.write(f"- **Files**: {structure['file_count']}\n")
    f.write(f"- **Directories**: {structure['dir_count']}\n")

    # Write important files section
    f.write("\n## Important Files\n\n")
    if important_files:
        for file_type, files in important_files.items():
            f.write(f"### {file_type.capitalize()}\n\n")
            for file in files:
                f.write(f"- `{file}`\n")
            f.write("\n")
    else:
        f.write("No important project files detected.\n\n")

    # Write project structure section
    f.write("\n## Project Structure\n\n")
    f.write("```\n")
    write_directory_tree(f, structure['structure'], project_name)
    f.write("```\n\n")

    # Write largest directories
    dir_sizes = [(path, data['size'], data['file_count'])
                 for path, data in structure['structure'].items()]
    dir_sizes.sort(key=lambda x: x[1], reverse=True)

    f.write("### Largest Directories\n\n")
    f.write("| Directory | Files | Size |\n")
    f.write("|-----------|-------|------|\n")
    for path, size, file_count in dir_sizes[:10]:  # Top 10
        display_path = path if path else project_name
        f.write(f"| {display_path} | {file_count} | {format_size(size)} |\n")
    f.write("\n")

    # Write dependency section
    f.write("\n## Dependencies\n\n")

    # Critical modules (most depended upon)
    if dependencies['critical']:
        f.write("### Critical Modules\n\n")
        f.write("These modules are most depended upon in the project:\n\n")
        f.write("| Module | Dependents |\n")
        f.write("|--------|------------|\n")
        for module in dependencies['critical']:
            f.write(f"| `{module['module']}` | {module['dependents']} |\n")
        f.write("\n")

    # Circular dependencies
    if dependencies['circular']:
        f.write("### Circular Dependencies\n\n")
        f.write("These modules form circular dependencies (potential design issue):\n\n")
        for i, cycle in enumerate(dependencies['circular']):
            f.write(f"{i + 1}. `{' → '.join(cycle)}`\n")
        f.write("\n")

    # Module relationships
    f.write("### Module Relationships\n\n")

    for module, deps in sorted(dependencies['dependencies'].items()):
        if deps:  # Only show modules with dependencies
            f.write(f"#### `{module}`\n\n")
            f.write("Depends on:\n")
            for dep in sorted(deps):
                f.write(f"- `{dep}`\n")

            # Show what depends on this module
            reverse_deps = dependencies['reverse_dependencies'].get(module, [])
            if reverse_deps:
                f.write("\nDepended on by:\n")
                for rev_dep in sorted(reverse_deps):
                    f.write(f"- `{rev_dep}`\n")

            f.write("\n")

    # Write documentation section
    f.write("\n## Documentation\n\n")

    if documentation:
        for file_path, docs in sorted(documentation.items()):
            f.write(f"### `{file_path}`\n\n")

            # Module docstring
            if docs['module_doc']:
                f.write(f"{docs['module_doc']}\n\n")

            # Classes
            if docs['classes']:
                f.write("#### Classes\n\n")
                for class_name, class_doc in docs['classes'].items():
                    f.write(f"##### `{class_name}`\n\n")
                    f.write(f"{class_doc}\n\n")

            # Functions
            if docs['functions']:
                f.write("#### Functions\n\n")
                for func_name, func_doc in docs['functions'].items():
                    f.write(f"##### `{func_name}`\n\n")
                    f.write(f"{func_doc}\n\n")
    else:
        f.write("No documentation found in the project.\n\n")

    # Summary and recommendations
    f.write("\n## Insights & Recommendations\n\n")

    # Add recommendations based on analysis
    recommendations = []

    # Check for circular dependencies
    if dependencies['circular']:
        recommendations.append("Resolve circular dependencies to improve maintainability.")

    # Check for documentation
    if not documentation:
        recommendations.append("Add docstrings to improve code documentation.")

    # Check for important files
    if 'readme' not in important_files:
        recommendations.append("Add a README file to document the project.")

    # Output recommendations
    if recommendations:
        for rec in recommendations:
            f.write(f"- {rec}\n")
    else:
        f.write("Project structure looks good with no obvious issues.\n")


def write_directory_tree(f, structure, project_name, path="", prefix=""):
    """Write a directory tree in text format."""
    if path == "":
        f.write(f"{project_name}/\n")

        # Get non-empty roots
        roots = sorted([p for p in structure.keys() if p and not '/' in p and not '\\' in p])

        # Process files in root
        root_data = structure.get("", {"files": []})
        for i, file_info in enumerate(sorted(root_data['files'], key=lambda x: x['name'])):
            is_last = (i == len(root_data['files']) - 1) and not roots
            f.write(f"├── {file_info['name']}\n" if not is_last else f"└── {file_info['name']}\n")

        # Process subdirectories
        for i, subdir in enumerate(roots):
            is_last = (i == len(roots) - 1)
            new_prefix = "└── " if is_last else "├── "
            f.write(f"{new_prefix}{os.path.basename(subdir)}/\n")

            child_prefix = "    " if is_last else "│   "
            subdir_data = structure.get(subdir, {"files": []})

            # Write files in the subdirectory
            for j, file_info in enumerate(sorted(subdir_data['files'], key=lambda x: x['name'])):
                is_last_file = (j == len(subdir_data['files']) - 1)
                file_prefix = "└── " if is_last_file else "├── "
                f.write(f"{child_prefix}{file_prefix}{file_info['name']}\n")

    else:
        # Handle deeper levels of nesting
        subdirs = [p for p in structure.keys() if p.startswith(path + "/")]

        # Files in current directory
        dir_data = structure.get(path, {"files": []})
        for i, file_info in enumerate(sorted(dir_data['files'], key=lambda x: x['name'])):
            is_last = (i == len(dir_data['files']) - 1) and not subdirs
            f.write(f"{prefix}{'└── ' if is_last else '├── '}{file_info['name']}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate system analysis report')
    parser.add_argument('--project-dir', '-p', default='.',
                        help='Path to the project directory (default: current directory)')
    parser.add_argument('--output', '-o', default='system_analysis.md',
                        help='Output file path (default: system_analysis.md)')
    parser.add_argument('--exclude', '-e', nargs='+',
                        help='Additional directories to exclude from analysis')

    args = parser.parse_args()

    # Validate project directory
    if not os.path.isdir(args.project_dir):
        print(f"Error: {args.project_dir} is not a valid directory")
        sys.exit(1)

    # Prepare exclude directories
    exclude_dirs = ['venv', 'env', '.git', '__pycache__', 'node_modules', '.idea', '.vscode', 'build', 'dist']
    if args.exclude:
        exclude_dirs.extend(args.exclude)

    # Run analysis
    analyze_system(args.project_dir, args.output, exclude_dirs)


if __name__ == "__main__":
    main()