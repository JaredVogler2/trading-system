"""Simple script to bundle all Python files."""

from pathlib import Path
import datetime

# Get all Python files
root = Path(".")
output_file = f"code_bundle_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

ignore = {'.venv', '__pycache__', 'venv', '.git'}

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("TRADING SYSTEM CODE BUNDLE\n")
    f.write(f"Generated: {datetime.datetime.now()}\n")
    f.write("=" * 80 + "\n\n")

    for py_file in sorted(root.rglob("*.py")):
        if not any(ig in str(py_file) for ig in ignore):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"FILE: {py_file}\n")
            f.write(f"{'=' * 80}\n\n")
            try:
                f.write(py_file.read_text(encoding='utf-8'))
                f.write("\n")
            except Exception as e:
                f.write(f"Error reading file: {e}\n")

print(f"Bundle created: {output_file}")