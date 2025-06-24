import ast
import os
import re
from pathlib import Path
import json
from collections import defaultdict, deque


class MultiEntryDependencyAnalyzer:
    def __init__(self, project_path):
        self.project_path = Path(project_path).resolve()
        self.all_py_files = {}  # {relative_path: absolute_path}
        self.imports_map = defaultdict(set)  # {file: set of imported files}
        self.dynamic_calls = defaultdict(set)  # {file: set of dynamically called files}
        self.reverse_imports = defaultdict(set)  # {file: set of files that import it}
        self.entry_points = {}  # {entry_point: set of reachable files}

    def find_all_py_files(self):
        """Find all .py files in the project."""
        for root, dirs, files in os.walk(self.project_path):
            # Skip virtual environments and cache directories
            dirs[:] = [d for d in dirs if
                       d not in {'.venv', '__pycache__', '.git', 'venv', 'env', '.tox', 'build', 'dist'}]

            for file in files:
                if file.endswith('.py'):
                    abs_path = Path(root) / file
                    rel_path = abs_path.relative_to(self.project_path)
                    self.all_py_files[str(rel_path)] = abs_path

    def find_dynamic_executions(self, file_path):
        """Find subprocess calls, exec statements, and other dynamic executions."""
        dynamic_files = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for subprocess patterns
            subprocess_patterns = [
                r'subprocess\.(?:run|call|Popen)\s*\(\s*\[?\s*["\']python["\']?\s*,\s*["\']([^"\']+\.py)["\']',
                r'subprocess\.(?:run|call|Popen)\s*\(\s*["\']python\s+([^"\']+\.py)["\']',
                r'os\.system\s*\(\s*["\']python\s+([^"\']+\.py)',
            ]

            for pattern in subprocess_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    # Clean up the match and try to resolve to a project file
                    cleaned = match.strip()
                    if '/' in cleaned or '\\' in cleaned:
                        # It's a path
                        potential_path = Path(cleaned)
                        if potential_path.name in [Path(f).name for f in self.all_py_files]:
                            for rel_path in self.all_py_files:
                                if Path(rel_path).name == potential_path.name:
                                    dynamic_files.add(rel_path)
                    else:
                        # Just a filename
                        for rel_path in self.all_py_files:
                            if Path(rel_path).name == cleaned:
                                dynamic_files.add(rel_path)

            # Look for specific dashboard references
            if 'enhanced_dashboard' in content or 'run_enhanced_dashboard' in content:
                # Try to find the dashboard file
                for rel_path in self.all_py_files:
                    if 'enhanced_dashboard' in rel_path:
                        dynamic_files.add(rel_path)

            # Look for threading/multiprocessing that might launch other scripts
            if 'Thread' in content or 'Process' in content:
                # Simple heuristic - look for .py files mentioned in the same context
                for rel_path in self.all_py_files:
                    if Path(rel_path).stem in content:
                        dynamic_files.add(rel_path)

        except Exception as e:
            print(f"Error analyzing dynamic calls in {file_path}: {e}")

        return dynamic_files

    def parse_imports(self, file_path):
        """Extract import statements from a Python file."""
        imports = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        if node.level == 0:  # absolute import
                            imports.add(node.module)
                        else:  # relative import
                            current_package = str(file_path.relative_to(self.project_path).parent)
                            if current_package == '.':
                                current_package = ''

                            if node.level == 1:  # from . import
                                if current_package:
                                    imports.add(current_package)
                            else:  # from .. import or deeper
                                parts = current_package.split('/')
                                if len(parts) >= node.level - 1:
                                    parent_package = '/'.join(parts[:-(node.level - 1)])
                                    if node.module:
                                        imports.add(
                                            f"{parent_package}.{node.module}" if parent_package else node.module)
                                    elif parent_package:
                                        imports.add(parent_package)

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

        return imports

    def resolve_import_to_file(self, import_name, importing_file):
        """Resolve an import name to an actual file in the project."""
        resolved_files = set()

        # Convert module path to potential file paths
        parts = import_name.replace('.', '/')

        # Check for direct file match
        potential_paths = [
            f"{parts}.py",
            f"{parts}/__init__.py",
        ]

        # Also check if it's importing from a package
        parts_list = import_name.split('.')
        for i in range(len(parts_list)):
            partial_path = '/'.join(parts_list[:i + 1])
            potential_paths.extend([
                f"{partial_path}.py",
                f"{partial_path}/__init__.py"
            ])

        for potential in potential_paths:
            if potential in self.all_py_files:
                resolved_files.add(potential)

        return resolved_files

    def build_dependency_graph(self):
        """Build the complete import and dynamic call dependency graph."""
        print("Building dependency graph...")

        for rel_path, abs_path in self.all_py_files.items():
            # Parse static imports
            imports = self.parse_imports(abs_path)

            for import_name in imports:
                resolved_files = self.resolve_import_to_file(import_name, rel_path)
                for resolved in resolved_files:
                    self.imports_map[rel_path].add(resolved)
                    self.reverse_imports[resolved].add(rel_path)

            # Find dynamic executions
            dynamic_files = self.find_dynamic_executions(abs_path)
            for dynamic in dynamic_files:
                self.dynamic_calls[rel_path].add(dynamic)

    def find_reachable_files(self, entry_point):
        """Find all files reachable from an entry point."""
        if entry_point not in self.all_py_files:
            print(f"Entry point {entry_point} not found!")
            return set()

        visited = set()
        queue = deque([entry_point])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue

            visited.add(current)

            # Add statically imported files
            for imported in self.imports_map.get(current, []):
                if imported not in visited:
                    queue.append(imported)

            # Add dynamically called files
            for dynamic in self.dynamic_calls.get(current, []):
                if dynamic not in visited:
                    queue.append(dynamic)

        return visited

    def analyze_multiple_entry_points(self, entry_points):
        """Analyze dependencies from multiple entry points."""
        self.find_all_py_files()
        self.build_dependency_graph()

        # Find reachable files from each entry point
        for entry in entry_points:
            self.entry_points[entry] = self.find_reachable_files(entry)

        # Calculate union of all reachable files
        all_used = set()
        for files in self.entry_points.values():
            all_used.update(files)

        # Find unused files
        unused_files = set(self.all_py_files.keys()) - all_used

        return all_used, unused_files

    def generate_report(self, entry_points):
        """Generate a detailed report for multiple entry points."""
        all_used, unused_files = self.analyze_multiple_entry_points(entry_points)

        report = []
        report.append(f"Multi-Entry Point Dependency Analysis")
        report.append(f"Project: {self.project_path}")
        report.append(f"Entry Points: {', '.join(entry_points)}")
        report.append("=" * 80)
        report.append("")

        # Summary
        report.append(f"Total Python files: {len(self.all_py_files)}")
        report.append(f"Files used by any entry point: {len(all_used)}")
        report.append(f"Files not used by any entry point: {len(unused_files)}")
        report.append("")

        # Per entry point analysis
        for entry in entry_points:
            reachable = self.entry_points.get(entry, set())
            report.append(f"\nFiles reachable from {entry}: {len(reachable)}")

            # Check if one entry point calls another
            for other_entry in entry_points:
                if other_entry != entry and other_entry in reachable:
                    report.append(f"  [CONNECTED] {entry} connects to {other_entry}")

                    # Show how they're connected
                    if other_entry in self.imports_map.get(entry, set()):
                        report.append(f"    - via static import")
                    if other_entry in self.dynamic_calls.get(entry, set()):
                        report.append(f"    - via dynamic execution (subprocess/exec)")

        # Files used by both
        if len(entry_points) > 1:
            shared_files = set.intersection(*self.entry_points.values())
            report.append(f"\nFiles used by ALL entry points: {len(shared_files)}")
            if len(shared_files) < 20:  # Only show if reasonable number
                for f in sorted(shared_files):
                    report.append(f"  - {f}")

        # Show dynamic calls found
        if any(self.dynamic_calls.values()):
            report.append("\nDYNAMIC EXECUTIONS FOUND:")
            report.append("-" * 40)
            for file, called in self.dynamic_calls.items():
                if called:
                    report.append(f"{file} dynamically calls:")
                    for c in called:
                        report.append(f"  -> {c}")

        # Categorize unused files
        report.append("\nUNUSED FILES (not reachable from any entry point):")
        report.append("-" * 40)

        test_files = []
        setup_files = []
        other_entry_points = []
        truly_unused = []

        for file in sorted(unused_files):
            if 'test' in file.lower():
                test_files.append(file)
            elif 'setup' in file or 'install' in file or 'config' in file:
                setup_files.append(file)
            elif file.startswith('run_') or file in ['main.py']:
                other_entry_points.append(file)
            else:
                truly_unused.append(file)

        if test_files:
            report.append("\n  Test Files:")
            for f in test_files:
                size = self.all_py_files[f].stat().st_size
                report.append(f"    - {f} ({size:,} bytes)")

        if setup_files:
            report.append("\n  Setup/Configuration Scripts:")
            for f in setup_files:
                size = self.all_py_files[f].stat().st_size
                report.append(f"    - {f} ({size:,} bytes)")

        if other_entry_points:
            report.append("\n  Other Potential Entry Points:")
            for f in other_entry_points:
                size = self.all_py_files[f].stat().st_size
                report.append(f"    - {f} ({size:,} bytes)")

        if truly_unused:
            report.append("\n  Likely Unused Files:")
            for f in truly_unused:
                size = self.all_py_files[f].stat().st_size
                report.append(f"    - {f} ({size:,} bytes)")

        # Save reports
        report_text = '\n'.join(report)
        with open('multi_entry_dependency_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        # Save detailed data
        analysis_data = {
            'entry_points': entry_points,
            'total_files': len(self.all_py_files),
            'all_used_files': sorted(list(all_used)),
            'unused_files': sorted(list(unused_files)),
            'per_entry_point': {
                entry: sorted(list(files)) for entry, files in self.entry_points.items()
            },
            'import_graph': {k: sorted(list(v)) for k, v in self.imports_map.items()},
            'dynamic_calls': {k: sorted(list(v)) for k, v in self.dynamic_calls.items()},
            'categorized_unused': {
                'test_files': test_files,
                'setup_files': setup_files,
                'other_entry_points': other_entry_points,
                'likely_unused': truly_unused
            }
        }

        with open('multi_entry_dependency_data.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2)

        print(report_text)
        print(f"\nReports saved to:")
        print(f"  - multi_entry_dependency_report.txt")
        print(f"  - multi_entry_dependency_data.json")

        return truly_unused


def main():
    project_path = r"C:\Users\jared\PycharmProjects\trading_system"
    analyzer = MultiEntryDependencyAnalyzer(project_path)

    # Analyze both entry points
    entry_points = ["run_always_on_trading.py", "run_enhanced_dashboard.py"]

    print(f"Analyzing dependencies from multiple entry points: {', '.join(entry_points)}")
    print("This will find files used by either or both programs...")
    print()

    truly_unused = analyzer.generate_report(entry_points)

    if truly_unused:
        print(f"\n⚠️  Found {len(truly_unused)} likely unused files")
        print("These files are not reachable from any of the specified entry points")
        print("Review carefully before deleting!")


if __name__ == "__main__":
    main()