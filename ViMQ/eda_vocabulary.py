import json
import pandas as pd
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import os
import requests # To download the stop words list
from pyvi import ViTokenizer

# --- This script assumes the following directory structure ---
# your_project_folder/
# │
# ├── this_script.py
# │
# └── data/
#     ├── train.json
#     ├── dev.json
#     └── test.json
# -----------------------------------------------------------


# --- 1. Load the Data ---
def load_data(file_path):
    """Loads a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        print("Please make sure your data files are in a 'data' subfolder.")
        exit() # Exit the script if data is not found

# --- Setup Paths and Load Data ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

train_data = load_data(os.path.join(data_dir, 'train.json'))
dev_data = load_data(os.path.join(data_dir, 'dev.json'))
test_data = load_data(os.path.join(data_dir, 'test.json'))

# Combine all datasets for a complete vocabulary analysis
all_data = train_data + dev_data + test_data
df = pd.DataFrame(all_data)

print(f"Successfully loaded {len(all_data)} total samples from the dataset.")

# --- 2. Vocabulary and Word Frequency Analysis (using pyvi) ---
print("\n--- Running Vocabulary Analysis using pyvi ---")

# --- Step 1: Get a reliable Vietnamese stop words list ---
# This part remains the same. We download a standard list.
try:
    url = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt"
    response = requests.get(url)
    response.raise_for_status() # Raises an exception for bad status codes
    vietnamese_stopwords = set(response.text.splitlines())
    
    my_custom_filter_words = {"điều_trị", "dấu_hiệu", "bệnh", "thuốc", "làm_sao", "uống", "nguy_hiểm", "có_thể", "thế_nào",
                              "kèm", "như_thế_nào", "ảnh_hưởng", "đau", "mắc", "liệu", "bệnh_lý", "tuần", "xét_nghiệm",
                              "xuất_hiện"}

    # Add your custom words to the main set
    vietnamese_stopwords.update(my_custom_filter_words)
    
    print(f"Successfully loaded {len(vietnamese_stopwords)} Vietnamese stop words.")
except requests.exceptions.RequestException as e:
    print(f"Error downloading stop words: {e}")
    vietnamese_stopwords = set() # Use an empty set if download fails

# --- Step 2: Use pyvi for proper word tokenization ---
# Combine all sentences into a single corpus
corpus = ' '.join(df['sentence'])

# ViTokenizer.tokenize() returns a string with words correctly segmented by underscores
segmented_corpus = ViTokenizer.tokenize(corpus).lower()
words = segmented_corpus.split() # Now we can safely split by spaces

# --- Step 3: Filter out stop words and punctuation ---
# We check if the word is in our stopword list or if it's just punctuation
# We use replace('_', '') to check if the token consists only of letters
filtered_words = [
    word for word in words
    if word not in vietnamese_stopwords and word.replace('_', '').isalpha() and len(word) > 1
]

# --- Step 4: Perform the analysis ---
word_counts = Counter(filtered_words)

# --- 5. Display the Results ---
vocab_stats = (
    f"Total number of words (tokens) after filtering: {len(filtered_words)}\n"
    f"Vocabulary size (unique words) after filtering: {len(word_counts)}"
)
print("\n--- Vocabulary Statistics ---")
print(vocab_stats)

most_common_words = word_counts.most_common(20)
mcw_df = pd.DataFrame(most_common_words, columns=['Word', 'Count'])

print("\n--- 20 Most Common Words (after filtering) ---")
print(mcw_df.to_string(index=False))