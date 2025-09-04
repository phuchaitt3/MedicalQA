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
OUTPUT_DIR = os.path.join(script_dir, "eda_error_results")
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

df['word_count'] = df['sentence'].apply(lambda x: len(x.split()))
df['char_count'] = df['sentence'].apply(len)

write_to_report("EDA Survey for ViMQ Dataset", title="START OF REPORT")

# --- 2a. Data Quality and Anomaly Detection ---
report_title = "2a. Data Quality and Anomaly Detection"
write_to_report("", title=report_title) # Section separator

# --- Missing Value Detection ---
# Check for null/None values in the main columns of the DataFrame
missing_values = df[['sentence', 'sent_label', 'seq_label']].isnull().sum()
missing_values_report = (
    f"Missing values check:\n{missing_values}\n\n"
    f"Percentage of missing values:\n{(missing_values / len(df)) * 100}"
)
write_to_report(missing_values_report, title="Missing Value Analysis")

# --- Duplicate Sentence Detection ---
# Identify and count sentences that appear more than once
num_duplicates = df['sentence'].duplicated().sum()
duplicate_report = f"Number of duplicate sentences found: {num_duplicates}"
write_to_report(duplicate_report, title="Duplicate Sentence Analysis")
if num_duplicates > 0:
    # If duplicates exist, show a few examples in the report for inspection
    duplicate_examples = df[df['sentence'].duplicated(keep=False)].sort_values('sentence').head(6)
    write_to_report(duplicate_examples[['sentence', 'sent_label']], title="Examples of Duplicate Sentences")

# --- Outlier Detection in Sentence Length ---
# Generate box plots for a clear visual representation of outliers
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(y=df['word_count'], ax=axes[0])
axes[0].set_title('Outlier Detection in Word Count')
axes[0].set_ylabel('Number of Words')
sns.boxplot(y=df['char_count'], ax=axes[1], color='salmon')
axes[1].set_title('Outlier Detection in Character Count')
axes[1].set_ylabel('Number of Characters')
plt.suptitle('Sentence Length Outlier Analysis (Box Plots)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'sentence_length_outliers.png'), bbox_inches='tight')
plt.close()
print("Saved: sentence_length_outliers.png")

# Identify outliers statistically using the IQR (Interquartile Range) method
outlier_report_lines = []
for col in ['word_count', 'char_count']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the DataFrame to find data points outside the calculated bounds
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_report_lines.append(f"Identified {len(outliers)} outliers in '{col}' using the IQR method.")
    outlier_report_lines.append(f"  - Lower bound (Q1 - 1.5*IQR): {lower_bound:.2f}")
    outlier_report_lines.append(f"  - Upper bound (Q3 + 1.5*IQR): {upper_bound:.2f}")
    if not outliers.empty:
        # Show a few examples of outlier sentences in the report
        outlier_report_lines.append("  - Example outlier sentences (first 3):")
        for index, row in outliers.head(3).iterrows():
            outlier_report_lines.append(f"    - Length: {row[col]}, Sentence: '{row['sentence'][:100]}...'")

write_to_report("\n".join(outlier_report_lines), title="Statistical Outlier Identification (IQR Method)")


# --- NER Label Integrity Check ---
# Verify that entity labels in the data match the official entity set
all_entities_in_data = set([entity[2] for entry in df['seq_label'] for entity in entry])
official_entity_set = set(entity_labels)

# Find labels that are in the data but not in the official list (potential typos or errors)
undefined_labels = all_entities_in_data - official_entity_set
# Find labels that are in the official list but never used in the data
unused_official_labels = official_entity_set - all_entities_in_data

label_integrity_report = (
    f"Total unique entity labels found in data: {len(all_entities_in_data)}\n"
    f"Total unique entity labels in official set: {len(official_entity_set)}\n\n"
    f"Undefined labels (in data but not in official set): {undefined_labels or 'None'}\n"
    f"Unused official labels (in official set but not in data): {unused_official_labels or 'None'}"
)
write_to_report(label_integrity_report, title="NER Label Integrity Check")