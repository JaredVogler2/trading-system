# fix_news_analyzer.py
"""Quick fix for the timestamp issue in news analyzer."""

# Read the original file
with open('news/news_analyzer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the problematic line
old_code = """published = datetime.fromtimestamp(
                            pd.Timestamp(entry.published_parsed).timestamp()
                        )"""

new_code = """import time
                            published = datetime.fromtimestamp(
                                time.mktime(entry.published_parsed)
                            )"""

content = content.replace(old_code, new_code)

# Save the fixed version
with open('news/news_analyzer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed news_analyzer.py!")