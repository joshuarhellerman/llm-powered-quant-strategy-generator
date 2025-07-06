#!/usr/bin/env python3
"""
Simple System Analyzer - Concise Report Version
"""

import os
import re
import sys
import ast
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='Generate system analysis report')
    parser.add_argument('--project-dir', '-p', default='.',
                        help='Path to the project directory (default: current directory)')
    parser.add_argument('--output', '-o', default='system_analysis.md',
                        help='Output file path (default: system_analysis.md)')
    parser.add_argument('--exclude', '-e', nargs='+',
                        help='Additional directories to exclude from analysis')

    args = parser.parse_args()

    project_dir = args.project_dir
    output_file = args.output

    if not os.path.isdir(project_dir):
        print(f"Error: {project_dir} is not a valid directory")
        sys.exit(1)

    exclude_dirs = ['venv', 'env', '.git', '__pycache__', 'node_modules', '.idea', '.vscode', 'build', 'dist']
    if args.exclude:
        exclude_dirs.extend(args.exclude)

    # ========== Collect Project Structure ==========
    structure = {}
    file_count = 0
    dir_count = 0

    for root, dirs, files in os.walk(project_dir):
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

    structure_summary = {
        'structure': structure,
        'file_count': file_count,
        'dir_count': dir_count
    }

    # ========== Extract Dependencies ==========
    dependencies = defaultdict(set)
    reverse_dependencies = defaultdict(set)
    python_files = []

    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_dir)
                python_files.append((rel_path, file_path))

    for rel_path, abs_path in python_files:
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            try:
                tree = ast.parse(content)
                imports = []
                from_imports = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            from_imports.append(node.module)

                for module in imports + from_imports:
                    parts = module.split('.')
                    for i in range(len(parts), 0, -1):
                        prefix = '.'.join(parts[:i])
                        for other_rel_path, _ in python_files:
                            other_module = os.path.splitext(other_rel_path)[0].replace('/', '.').replace('\\', '.')
                            if other_module == prefix or other_module.endswith('.' + prefix):
                                dependencies[rel_path].add(other_rel_path)
                                reverse_dependencies[other_rel_path].add(rel_path)
                                break
            except SyntaxError:
                import_pattern = r'^(?:from\s+(\S+)\s+import|import\s+(\S+))'
                for line in content.splitlines():
                    m = re.match(import_pattern, line)
                    if m:
                        module = m.group(1) or m.group(2)
                        dependencies[rel_path].add(module)
        except:
            pass

    circular = []
    visited = set()

    def find_circular(node, path=None):
        if path is None:
            path = []

        if node in path:
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

    critical = []
    for module, deps in reverse_dependencies.items():
        if len(deps) >= 3:
            critical.append({'module': module, 'dependents': len(deps)})
    critical.sort(key=lambda x: x['dependents'], reverse=True)

    dependencies_summary = {
        'dependencies': {k: list(v) for k, v in dependencies.items()},
        'reverse_dependencies': {k: list(v) for k, v in reverse_dependencies.items()},
        'circular': circular[:5],  # Concise: limit to 5 cycles
        'critical': critical[:5]   # Concise: limit to top 5 critical modules
    }

    # ========== Find Important Files ==========
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
            for file_type, patterns in important_patterns.items():
                if any(re.match(pattern, file, re.IGNORECASE) for pattern in patterns):
                    rel_path = os.path.relpath(os.path.join(root, file), project_dir)
                    if file_type not in important:
                        important[file_type] = []
                    important[file_type].append(rel_path)

    # ========== Extract Documentation ==========
    docs = {}

    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_dir)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    try:
                        tree = ast.parse(content)
                        module_doc = ast.get_docstring(tree)
                        # Skip detailed class/function docs for brevity
                        if module_doc:
                            docs[rel_path] = {'module_doc': module_doc}
                    except SyntaxError:
                        pass
                except:
                    pass

    # ========== Helper for formatting size ==========
    def format_size(size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    # ========== Helper for writing directory tree ==========
    def write_directory_tree(f, structure, project_name, path="", prefix=""):
        if path == "":
            f.write(f"{project_name}/\n")
            roots = sorted([p for p in structure.keys() if p and not '/' in p and not '\\' in p])
            root_data = structure.get("", {"files": []})
            for i, file_info in enumerate(sorted(root_data['files'], key=lambda x: x['name'])):
                is_last = (i == len(root_data['files']) - 1) and not roots
                f.write(f"├── {file_info['name']}\n" if not is_last else f"└── {file_info['name']}\n")
            for i, subdir in enumerate(roots):
                is_last = (i == len(roots) - 1)
                new_prefix = "└── " if is_last else "├── "
                f.write(f"{new_prefix}{os.path.basename(subdir)}/\n")
                child_prefix = "    " if is_last else "│   "
                subdir_data = structure.get(subdir, {"files": []})
                for j, file_info in enumerate(sorted(subdir_data['files'], key=lambda x: x['name'])):
                    is_last_file = (j == len(subdir_data['files']) - 1)
                    file_prefix = "└── " if is_last_file else "├── "
                    f.write(f"{child_prefix}{file_prefix}{file_info['name']}\n")
        else:
            subdirs = [p for p in structure.keys() if p.startswith(path + "/")]
            dir_data = structure.get(path, {"files": []})
            for i, file_info in enumerate(sorted(dir_data['files'], key=lambda x: x['name'])):
                is_last = (i == len(dir_data['files']) - 1) and not subdirs
                f.write(f"{prefix}{'└── ' if is_last else '├── '}{file_info['name']}\n")

    # ========== Write the report ==========
    project_name = os.path.basename(os.path.abspath(project_dir))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# System Analysis: {project_name}\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Project Directory**: {project_dir}\n")
        f.write(f"- **Files**: {structure_summary['file_count']}\n")
        f.write(f"- **Directories**: {structure_summary['dir_count']}\n\n")

        f.write("## Important Files\n\n")
        if important:
            for file_type, files in important.items():
                f.write(f"### {file_type.capitalize()}\n\n")
                for file in files[:3]:  # Concise: show only first 3 files per type
                    f.write(f"- `{file}`\n")
                if len(files) > 3:
                    f.write(f"- ...and {len(files) - 3} more\n")
                f.write("\n")
        else:
            f.write("No important project files detected.\n\n")

        f.write("## Project Structure\n\n")
        f.write("```\n")
        write_directory_tree(f, structure_summary['structure'], project_name)
        f.write("```\n\n")

        dir_sizes = [(path, data['size'], data['file_count'])
                     for path, data in structure_summary['structure'].items()]
        dir_sizes.sort(key=lambda x: x[1], reverse=True)

        f.write("### Largest Directories\n\n")
        f.write("| Directory | Files | Size |\n")
        f.write("|-----------|-------|------|\n")
        for path, size, file_cnt in dir_sizes[:5]:  # Concise: limit to top 5
            display_path = path if path else project_name
            f.write(f"| {display_path} | {file_cnt} | {format_size(size)} |\n")
        f.write("\n")

        f.write("## Dependencies\n\n")
        if dependencies_summary['critical']:
            f.write("### Critical Modules\n\n")
            f.write("Most depended upon modules:\n\n")
            f.write("| Module | Dependents |\n")
            f.write("|--------|------------|\n")
            for module in dependencies_summary['critical']:
                f.write(f"| `{module['module']}` | {module['dependents']} |\n")
            f.write("\n")

        if dependencies_summary['circular']:
            f.write("### Circular Dependencies\n\n")
            f.write("Potential design issues:\n\n")
            for i, cycle in enumerate(dependencies_summary['circular']):
                f.write(f"{i + 1}. `{' → '.join(cycle)}`\n")
            f.write("\n")

        f.write("### Module Relationships (Summary)\n\n")
        for module in sorted(dependencies_summary['dependencies'].keys()):
            f.write(f"- `{module}`\n")
        f.write("\n")

        f.write("## Documentation\n\n")
        if docs:
            for file_path, d in sorted(docs.items()):
                f.write(f"### `{file_path}`\n\n")
                f.write(f"{d['module_doc']}\n\n")
        else:
            f.write("No module-level docstrings found.\n\n")

        f.write("## Insights & Recommendations\n\n")
        recommendations = []
        if dependencies_summary['circular']:
            recommendations.append("Resolve circular dependencies to improve maintainability.")
        if not docs:
            recommendations.append("Add module-level docstrings to improve code documentation.")
        if 'readme' not in important:
            recommendations.append("Add a README file to document the project.")

        for rec in recommendations[:3]:  # Concise: max 3 recommendations
            f.write(f"- {rec}\n")
        if not recommendations:
            f.write("Project structure looks good with no obvious issues.\n")

    print(f"Analysis complete. Report saved to: {output_file}")


if __name__ == "__main__":
    main()
