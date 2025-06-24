import os
from pathlib import Path


def count_lines_in_file(filepath):
    """Count the number of lines in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0


def scan_python_files(directory):
    """Scan directory for .py files and count lines in each."""
    directory_path = Path(directory)

    if not directory_path.exists():
        print(f"Directory not found: {directory}")
        return

    if not directory_path.is_dir():
        print(f"Path is not a directory: {directory}")
        return

    # Dictionary to store results
    py_files = {}
    total_lines = 0

    # Walk through directory and subdirectories
    for root, dirs, files in os.walk(directory_path):
        # Skip .venv directory
        if '.venv' in dirs:
            dirs.remove('.venv')

        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                lines = count_lines_in_file(filepath)

                # Store relative path from base directory
                relative_path = filepath.relative_to(directory_path)
                py_files[str(relative_path)] = lines
                total_lines += lines

    # Display results
    if not py_files:
        print("No Python files found in the specified directory.")
        return

    print(f"Python files in: {directory}")
    print("-" * 70)
    print(f"{'File Path':<50} {'Lines':>10}")
    print("-" * 70)

    # Sort files by path
    for filepath in sorted(py_files.keys()):
        print(f"{filepath:<50} {py_files[filepath]:>10}")

    print("-" * 70)
    print(f"{'Total files: ' + str(len(py_files)):<50} {total_lines:>10}")
    print("-" * 70)

    # Optional: Save results to a text file
    output_file = "python_files_line_count.txt"
    with open(output_file, 'w') as f:
        f.write(f"Python files in: {directory}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'File Path':<50} {'Lines':>10}\n")
        f.write("-" * 70 + "\n")

        for filepath in sorted(py_files.keys()):
            f.write(f"{filepath:<50} {py_files[filepath]:>10}\n")

        f.write("-" * 70 + "\n")
        f.write(f"{'Total files: ' + str(len(py_files)):<50} {total_lines:>10}\n")

    print(f"\nResults saved to: {output_file}")


# Main execution
if __name__ == "__main__":
    folder_path = r"C:\Users\jared\PycharmProjects\trading_system"
    scan_python_files(folder_path)