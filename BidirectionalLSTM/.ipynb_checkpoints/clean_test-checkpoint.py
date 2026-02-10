import re

# Load the file
with open('test_output.txt', 'r') as f:
    content = f.read()

# 1. Remove the tags correctly
# We use r'\[...\]' to escape the square brackets which are special in regex
clean_content = re.sub(r"\", "", content)

# 2. Split by lines and remove empty ones
lines = [line.strip() for line in clean_content.split('\n') if line.strip()]

# 3. Save the final clean version (one translation per line)
with open('cleaned_output.txt', 'w') as f:
    for line in lines:
        f.write(line + '\n')

print(f"Cleaned {len(lines)} lines. Now run sacrebleu!")