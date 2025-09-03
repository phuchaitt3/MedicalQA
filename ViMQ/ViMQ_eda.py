import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import os
import requests # To download the stop words list
from pyvi import ViTokenizer

# --- Setup Output Directory ---
script_dir = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(script_dir, "eda_results")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
REPORT_FILE = os.path.join(OUTPUT_DIR, "eda_report.txt")

# Clear the report file if it exists
if os.path.exists(REPORT_FILE):
    os.remove(REPORT_FILE)

def write_to_report(content, title=""):
    """Appends content to the report file."""
    with open(REPORT_FILE, 'a', encoding='utf-8') as f:
        if title:
            f.write(f"--- {title} ---\n")
        f.write(str(content) + "\n\n")

# --- 1. Load the Data ---
def load_data(file_path):
    """Loads a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_entity_set(file_path):
    """Loads the entity set from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

# Define data paths (assuming a 'data' subfolder)
train_data = load_data(os.path.join(script_dir, 'data', 'train.json'))
dev_data = load_data(os.path.join(script_dir, 'data', 'dev.json'))
test_data = load_data(os.path.join(script_dir, 'data', 'test.json'))
all_data = train_data + dev_data + test_data

entity_labels = load_entity_set(os.path.join(script_dir, 'data', 'entity_set.txt'))
df = pd.DataFrame(all_data)

write_to_report("EDA Survey for ViMQ Dataset", title="START OF REPORT")

# --- 2. Basic Data and Sentence Analysis ---
report_title = "2. Basic Data and Sentence Analysis"
basic_stats = (
    f"Number of training samples: {len(train_data)}\n"
    f"Number of development samples: {len(dev_data)}\n"
    f"Number of test samples: {len(test_data)}\n"
    f"Total number of samples: {len(all_data)}"
)
write_to_report(basic_stats, title=report_title)

df['word_count'] = df['sentence'].apply(lambda x: len(x.split()))
df['char_count'] = df['sentence'].apply(len)
write_to_report(df[['word_count', 'char_count']].describe(), title="Sentence Length Statistics")

# Plot and save sentence length distributions
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(df['word_count'], bins=30, ax=axes[0], kde=True)
axes[0].set_title('Distribution of Sentence Length (by Word Count)')
axes[0].set_xlabel('Number of Words')
axes[0].set_ylabel('Frequency')
sns.histplot(df['char_count'], bins=30, ax=axes[1], kde=True, color='salmon')
axes[1].set_title('Distribution of Sentence Length (by Character Count)')
axes[1].set_xlabel('Number of Characters')
axes[1].set_ylabel('Frequency')
plt.suptitle('Sentence Length Analysis', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'sentence_length_distribution.png'), bbox_inches='tight')
plt.close()
print("Saved: sentence_length_distribution.png")

# --- 3. Intent Classification Analysis (`sent_label`) ---
report_title = "3. Intent Classification Analysis"
intent_counts = df['sent_label'].value_counts()
write_to_report(intent_counts, title=report_title)

# Plot and save intent distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=intent_counts.index, y=intent_counts.values, hue=intent_counts.index, palette='viridis', legend=False)
plt.title('Distribution of Intent Labels')
plt.xlabel('Intent Label')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_DIR, 'intent_distribution.png'), bbox_inches='tight')
plt.close()
print("Saved: intent_distribution.png")

# --- 4. Named Entity Recognition (NER) Analysis (`seq_label`) ---
report_title = "4. Named Entity Recognition (NER) Analysis"
all_entities = [entity[2] for entry in df['seq_label'] for entity in entry]
entity_counts = Counter(all_entities)
write_to_report(pd.Series(entity_counts), title=report_title)

# Plot and save entity distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=list(entity_counts.keys()), y=list(entity_counts.values()), palette='plasma', hue=list(entity_counts.keys()))
plt.title('Distribution of Entity Types (NER)')
plt.xlabel('Entity Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_DIR, 'entity_type_distribution.png'), bbox_inches='tight')
plt.close()
print("Saved: entity_type_distribution.png")

# Analyze and save number of entities per sentence
df['num_entities'] = df['seq_label'].apply(len)
write_to_report(df['num_entities'].describe(), title="Statistics for Number of Entities per Sentence")

plt.figure(figsize=(10, 6))
sns.countplot(x='num_entities', data=df, hue='num_entities', palette='magma', legend=False)
plt.title('Number of Entities per Sentence')
plt.xlabel('Number of Entities')
plt.ylabel('Count of Sentences')
plt.savefig(os.path.join(OUTPUT_DIR, 'entities_per_sentence.png'), bbox_inches='tight')
plt.close()
print("Saved: entities_per_sentence.png")

# --- 5. Vocabulary and Word Frequency Analysis ---
report_title = "5. Vocabulary and Word Frequency Analysis"
try:
    url = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt"
    response = requests.get(url)
    response.raise_for_status() # Raises an exception for bad status codes
    vietnamese_stopwords = set(response.text.splitlines())
    
    my_custom_filter_words = {"làm_sao", "có_thể", "thế_nào", "như_thế_nào", "liệu"}
    # "điều_trị", "dấu_hiệu", "bệnh", "thuốc", "uống", "nguy_hiểm", "kèm", "ảnh_hưởng", "đau", "mắc", "bệnh_lý", "tuần", "xét_nghiệm","xuất_hiện"

    # Add your custom words to the main set
    vietnamese_stopwords.update(my_custom_filter_words)
    
    print(f"Successfully loaded {len(vietnamese_stopwords)} Vietnamese stop words.")
except requests.exceptions.RequestException as e:
    print(f"Error downloading stop words: {e}")
    vietnamese_stopwords = set() # Use an empty set if download fails

corpus = ' '.join(df['sentence']).lower()
# ViTokenizer.tokenize() returns a string with words correctly segmented by underscores
segmented_corpus = ViTokenizer.tokenize(corpus).lower()
words = segmented_corpus.split()

# We check if the word is in our stopword list or if it's just punctuation
# We use replace('_', '') to check if the token consists only of letters
filtered_words = [
    word for word in words
    if word not in vietnamese_stopwords and word.replace('_', '').isalpha() and len(word) > 1
]

word_counts = Counter(filtered_words)
vocab_stats = (
    f"Total number of words (tokens) after filtering: {len(filtered_words)}\n"
    f"Vocabulary size (unique words) after filtering: {len(word_counts)}"
)
write_to_report(vocab_stats, title=report_title)

most_common_words = word_counts.most_common(20)
mcw_df = pd.DataFrame(most_common_words, columns=['word', 'count'])
write_to_report(mcw_df, title="20 Most Common Words (after filtering)")

# Plot and save the frequency of the most common words
plt.figure(figsize=(12, 8))
sns.barplot(x='count', y='word', data=mcw_df, hue='word', palette='cividis', legend=False)
plt.title('Top 20 Most Common Words in the Dataset')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.savefig(os.path.join(OUTPUT_DIR, 'top_20_words.png'), bbox_inches='tight')
plt.close()
print("Saved: top_20_words.png")

# --- 6. Combined Analysis: Intents vs. Entities ---
print("--- 6. Combined Analysis: Intents vs. Entities ---")
intent_entity_data = []
for index, row in df.iterrows():
    intent = row['sent_label']
    entities = [entity[2] for entity in row['seq_label']]
    if not entities:
        intent_entity_data.append({'intent': intent, 'entity_type': 'O'})
    else:
        for entity_type in entities:
            intent_entity_data.append({'intent': intent, 'entity_type': entity_type})

intent_entity_df = pd.DataFrame(intent_entity_data)
heatmap_data = pd.crosstab(intent_entity_df['intent'], intent_entity_df['entity_type'])

# Save the heatmap of the co-occurrence
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Co-occurrence of Intents and Entity Types')
plt.xlabel('Entity Type')
plt.ylabel('Intent Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'intent_entity_heatmap.png'), bbox_inches='tight')
plt.close()
print("Saved: intent_entity_heatmap.png")

print(f"\n--- EDA Survey Complete ---")
print(f"All results have been saved to the '{OUTPUT_DIR}' directory.")
print(f"Statistical report: {REPORT_FILE}")