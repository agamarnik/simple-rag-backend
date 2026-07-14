from pathlib import Path

# defining a list of document content (just some strings)
docs = ['document 1', 'file 2 with some text', 'sample Document  3']

# 1. Define the target folder and ensure it exists
output_dir = Path("./docs")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Loop to create multiple numbered files
for i, doc in enumerate(docs):
    # Construct the full file path (e.g., my_target_folder/file_1.txt)
    file_path = output_dir / f"file_{i}.txt"

    # Create the file and write text to it
    with open(file_path, "w") as file:
        # file.write(f"This is the content inside file number {i}.\n")
        file.write(doc) # enumerate lets us use doc instead of doc[i]

# 3. Loop to read files and count words
for i, doc in enumerate(docs):
    file_path = output_dir / f"file_{i}.txt"

    with open(file_path, "r") as file:
        content = file.read()
        print(content)
        words = content.split()  # Splits the string by whitespace into a list of words
        word_count = len(words)  # Count the words
        print(f"Word count in {file_path}: ", word_count)
