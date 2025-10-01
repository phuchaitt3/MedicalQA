#!/usr/bin/env python
# coding: utf-8

# - The max length of 180 is not a standard maximum length, but it is a common choice depending on the dataset and task. The standard max length for models like BERT is typically 512 tokens, which is the maximum sequence length that BERT was pre-trained with. However, using a shorter max length, like 180 tokens, is often done for one of the reasons:
#     - Shorter sequences require less memory and allow for larger batch sizes, especially when working with GPUs that have limited memory.

# In[2]:


# num_epochs = 1
# num_epochs_2nd = 1
# fast_run = False
# reduce_samples = True
# num_samples = 50
# batch_size = 32 # 32
# mask_percentage = 0.15  # Originally 0.5
# val_max_len = 180
# bool_evaluate = True
# bool_early_stop = True
# bool_1st_early_stop = True
# bool_zip = False


# In[3]:


num_epochs = 10 # Originally 5
num_epochs_2nd = 10
fast_run = False
reduce_samples = False
# num_samples = 1
batch_size = 32 # 32
mask_percentage = 0.15  # Originally 0.5
val_max_len = 180
bool_evaluate = True
bool_early_stop = True
bool_1st_early_stop = True
bool_zip = True


# In[5]:


import json
import emoji
from collections import Counter
import pandas as pd
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertForMaskedLM
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_scheduler


# # Import the JSON data and observe some entries
# 

# Open the JSON dataset.

# In[6]:


with open('/kaggle/input/hatexplain-dataset/HateXplain_dataset.json', 'r') as file:
    data = json.load(file)


# Four empty lists (post_ids, annotators, rationales, and post_tokens) are created. These will be used to store the data extracted from the JSON structure.

# In[7]:


post_ids = []
annotators = []
rationales = []
post_tokens = []


# The for loop iterates over each key-value pair (representing each post) in the JSON data and appends to the appropriate lists.

# In[8]:


for key, value in data.items():
    post_ids.append(value['post_id'])
    annotators.append(value['annotators'])
    rationales.append(value['rationales'])
    post_tokens.append(value['post_tokens'])


# Once the data is extracted from the JSON, it is organized into a Pandas DataFrame (df). 
# 
# The four lists (post_ids, annotators, rationales, and post_tokens) are converted into columns of the DataFrame, with each row representing a post from the dataset.

# In[9]:


df = pd.DataFrame({
    'post_id': post_ids,
    'annotators': annotators,
    'rationales': rationales,
    'post_tokens': post_tokens
})

# Display the DataFrame
df


# # Data Preprocessing

# In[10]:


# Initialize lists to store results
FINAL_LABEL_LIST = [] # Final labels for each post based on the majority vote among annotators.
input_data = [] # Stores the text content of each post
rationales = [] # Stores the rationale for each post
post_ids = [] # Stores the unique identifiers for each post
FINAL_TARGET_LIST = [] # Stores the final targets (e.g., communities or groups) for each post.


# - The Counter class from the collections module is used to count how many times each label appears in label_list, by creating a dictionary-like structure, where the keys are the labels and the values are the counts.
# - The total number of labels in the list is calculated using len(label_list).
# - The most_common(1) method of Counter returns a list of tuples, where each tuple contains a label and its count.
#     - The [0] index gets the first (most common) label, and this tuple is unpacked into most_common_label (the label) and count (how many times it appears).
# - To qualify as a majority, the most common label must appear more than 50% of the time.

# In[11]:


# Function to process the majority label
def get_majority_label(label_list):
    # Count the occurrences of each label in the list
    label_counts = Counter(label_list)
    total_labels = len(label_list)
    if total_labels == 0:
        return None
    # Find the most common label and its count
    most_common_label, count = label_counts.most_common(1)[0]
    # If the most common label appears more than 50% of the time, return it as the majority label
    if count / total_labels > 0.5:
        return most_common_label
    return None


# This code functions similarly to the majority label function but is adapted for handling targets. 
# 
# The key difference is that this function processes a list of annotators and their respective targets (which may involve multiple targets for each annotator), while the get_majority_label function processes a list of labels and finds a single majority label.
# 
# Example:
# - If 3 annotators provide the following targets for a post:
#     - Annotator 1: ['Women', 'African']
#     - Annotator 2: ['Women']
#     - Annotator 3: ['Women', 'Jewish', 'African']
# - The function will count:
#     - Women appears 3 times
#     - African appears 2 times
#     - Jewish appears 1 time
# - Greater than 0.5:
#     - Since Women appears in 3 out of 3 lists (100%), it is the majority target and will be returned.
#     - Since African appears in 2 out of 3 lists (66.7%), which is more than 50%, it is also considered a majority target and will be returned.
#     - Jewish appears only 1 time, which is less than 50%, so it won’t be considered a majority target.
# - The function will return: ['Women', 'African'].

# In[12]:


# Function to process the majority targets
def get_majority_targets(annotators):
    target_count = {}
    total_entries = len(annotators)
    if total_entries == 0:
        return ['None']
    # Iterate over each annotator's entry
    for entry in annotators:
        # Get the list of targets for the current annotator (default to an empty list if 'target' is not found)
        for target in entry.get('target', []):
            target_count[target] = target_count.get(target, 0) + 1 # Count and sum for each target
    # Find targets that appear in more than 50% of the total entries
    majority_targets = [target for target, count in target_count.items() if count / total_entries > 0.5]
    return majority_targets if majority_targets else ['None']


# The loop goes through each post in the data dictionary. The variable k represents the key (the post ID), and data[k] holds all the information related to that post:
# - The most common label (assigned_label) is appended to the FINAL_LABEL_LIST, which stores the majority labels for each post.
# - The majority targets for the post are appended to the FINAL_TARGET_LIST. 
# - The tokens from post_tokens are joined into a single string using ' '.join(). This turns the list of individual tokens back into a complete sentence or post. The resulting text is appended to the input_data list.
# - The post ID (data[k]['post_id']) is also appended to the post_ids list.
# - Rationales:
#     - **Special Case for Post '24439295_gab'**: This post has rationales of different lengths, so it is treated as a special case, and an empty list is appended to rationales for this post.
#     - **Averaging the Rationales**: If there are any rationales, the average value across all annotators is computed for each token using np.mean(rationales_array, axis=0). This averages the rationale annotations across the annotators for each token in the post.
#     - **Binarizing the Rationales**: The averaged rationales are then converted into binary values (1 or 0) using the condition 1 if value > 0.5 else 0. If the average value is greater than 0.5, it's considered important and labeled as 1, otherwise 0.
#     - **Handling Missing Rationales**: If there are no rationales for a post, a list of zeros with the length of post_tokens is created, indicating that no tokens are considered important.

# In[13]:


# Processing each post in data
for k in data.keys():
    
    # MAJORITY LABEL
    # Create a list of labels from the 'annotators' section of the current post
    label_list = [item['label'] for item in data[k]['annotators']]
    # Determine the most common label using the get_majority_label function
    assigned_label = get_majority_label(label_list)
    # Append the most common label to the FINAL_LABEL_LIST
    FINAL_LABEL_LIST.append(assigned_label)
    
    # MAJORITY TARGETS
    # Determine the majority targets using the get_majority_targets function
    majority_targets = get_majority_targets(data[k]['annotators'])
    # Append the majority targets to the FINAL_TARGET_LIST
    FINAL_TARGET_LIST.append(majority_targets)
    
    # POST TOKENS
    # Join the tokens in post_tokens into a single text string
    input_data.append(' '.join(data[k]['post_tokens']))
    
    # POST IDS
    # Append the post ID to the post_ids list
    post_ids.append(data[k]['post_id'])

    # RATIONALES
    if k == '24439295_gab': # Inner lists do not all have the same length
        rationales.append([])
    else:
        # Convert the list of rationales to a NumPy array for easier manipulation
        rationales_array = np.array(data[k]['rationales'])
        # If there are rationales, compute the mean and convert to binary labels
        if rationales_array.size > 0:
            # Compute the average of the rationales along axis 0
            averaged_rationales = np.mean(rationales_array, axis=0)
            # Convert the average values to binary labels (1 if > 0.5, otherwise 0)
            finalized_rationales = [1 if value > 0.5 else 0 for value in averaged_rationales]
        else:
            # If there are no rationales, create a list of zeros with the length of post_tokens
            finalized_rationales = [0] * len(data[k]['post_tokens'])
        rationales.append(finalized_rationales)


# Creating the DataFrame

# In[14]:


df = pd.DataFrame({
    'post_ids': post_ids,
    'input_text': input_data,
    'rationales': rationales,
    'label': FINAL_LABEL_LIST,
    'Final_target': FINAL_TARGET_LIST,
})


# This line removes any rows where the label column contains NaN (i.e., if there is no majority label for that post).
# 
# This is important because the model cannot be trained or evaluated on posts without a label.

# In[15]:


df = df.dropna(subset=['label'])


# [+] Fast run: Reduce number of samples

# In[16]:


if reduce_samples and num_samples < len(df):
    df = df.iloc[:num_samples].copy()


# After potentially reducing the dataset size or removing rows with NaN labels, the code extracts the updated values from the DataFrame back into lists.
# - input_data: A list of the input texts from the DataFrame’s input_text column.
# - encoded_labels: A list of the labels from the DataFrame’s label column.
# - rationales: A list of rationales from the DataFrame’s rationales column.
# 
# This ensures that the three lists (input_data, encoded_labels, rationales) remain in sync with the DataFrame and contain only the valid data.

# In[17]:


# H!split
input_data = df['input_text'].tolist()
encoded_labels = df['label'].tolist()
rationales = df['rationales'].tolist()

# Display the resulting DataFrame
print(df.head())
print(df['label'].value_counts())


# - Tokens:
#     - Earlier Joining: The tokens were joined into a string because it’s a common representation to store or work with complete texts.
#     - Current Splitting: The text is split back into tokens for token-level operations (like removing HTML tags and keeping rationales in sync).
# - Loop Through Tokens and Rationales:
#     - The function uses a regular expression (re.match(r'<.*?>', token)) to check if the token is an HTML tag (e.g., <'div'>, <'a'>, <'b'>).
#     - If the Token is Not an HTML Tag:
#         - It is added to the cleaned_tokens list.
#         - The corresponding rationale is also added to cleaned_rationales.
#     - If the Token is an allowed HTML Tag:
#         - The function checks if it is a special tag like <'br'> or <'hr'> using token.lower() in [<'br'>, <'hr'>].
#         - If it’s one of these allowed tags, the function keeps it and appends both the token and its rationale to the cleaned lists.
#         - These tags (<'br'> and <'hr'>) help maintain the visual or logical structure of the original text. In some cases, losing this structure can lead to misinterpretation, as important divisions between sentences or sections might be lost.
#     - For other HTML tags (e.g., <'div'>, <'a'>, etc.), the function skips both the tag and its corresponding rationale. This means that neither the tag nor its associated rationale will be included in the cleaned output.
# 

# In[18]:


def clean_html_tags_and_update_rationales(row):
    tokens = row['input_text'].split()
    rationales = row['rationales']
    cleaned_tokens = []
    cleaned_rationales = []

    for token, rationale in zip(tokens, rationales):
        # Check if the token is an HTML tag using regex
        if not re.match(r'<.*?>', token):
            cleaned_tokens.append(token)
            cleaned_rationales.append(rationale)
        else:
            if token.lower() in ['<br>', '<hr>']:  
                cleaned_tokens.append(token)
                cleaned_rationales.append(rationale)
            # Otherwise, skip the tag and its rationale

    # Join tokens back into a single string for input_text
    cleaned_text = ' '.join(cleaned_tokens)
    
    # Return the cleaned text and updated rationales
    return pd.Series([cleaned_text, cleaned_rationales], index=['input_text', 'rationales'])

# Apply the cleaning function to each row
df[['input_text', 'rationales']] = df.apply(clean_html_tags_and_update_rationales, axis=1)


# # Preprocessing target
# The point of the whole section of processing targets is to filter, clean, and refine the target communities associated with each post, ensuring that only communities of interest are retained for further analysis.

# This list, final_communities_sel, contains the communities of interest that we want to filter or focus on. Only these communities will be retained when processing the Final_target column.

# In[19]:


final_communities_sel = [
    'African', 'Islam', 'Jewish', 'Homosexual', 'Women', 'Refugee', 'Arab', 'Caucasian', 'Asian', 'Hispanic'
]


# Resetting the index ensures that the DataFrame's index is sequential (starting from 0) after any modifications (like row deletions or shuffling). The drop=True option ensures the old index is discarded.

# In[20]:


df.reset_index(inplace=True, drop=True)


# Loop Through Each Row in the DataFrame:
# - Check if Final_target is a list: The Final_target column for each post should be a list of target communities. This ensures that only lists are processed, and anything that isn’t a list is ignored or replaced with ['None'].
# - temp: Finds the intersection of the target communities for a post (from Final_target) and the predefined list of communities (final_communities_sel) and appends accordingly.

# In[21]:


final_target_information = []

for i in range(len(df)):
    # Check if the 'Final_target' column value is a list
    if isinstance(df['Final_target'][i], list):
        # Find the intersection of 'Final_target' values and the predefined communities of interest
        temp = list(set(df['Final_target'][i]) & set(final_communities_sel))
        
        if len(temp) == 0:
            # If there is no intersection (i.e., no relevant community), append 'None'
            final_target_information.append(['None'])
        else:
            # If there is an intersection, append the list of matching communities
            final_target_information.append(temp)
    else:
        # If 'Final_target' is not a list, append 'None' to maintain consistency
        final_target_information.append(['None'])
        
# Update the 'Final_target' column
df['Final_target'] = final_target_information


# The set(all_values) removes duplicates, creating a set of unique communities across all posts.

# In[22]:


all_values = [item for sublist in df['Final_target'] if isinstance(sublist, list) for item in sublist]
unique_values = set(all_values)

print("Unique Values:", unique_values)


# In[23]:


df


# # Fine-tuning 1: Fine-tuning BERT using MLM

# This code defines a custom PyTorch dataset class called TextDataset for a specific task where text data and their corresponding rationales (importance labels) are used for fine-tuning a BERT model. 
# 
# The dataset also includes logic to mask important tokens based on the provided rationales, and the class ensures that the data is properly tokenized and prepared for BERT. Additionally, the code handles issues like empty entries and applies token masking based on importance.
# 
# - The function **check_empty_entries()** is called to ensure no empty texts or rationales are present, which could cause issues during processing.
#     - When you create the TextDataset object, the __init__ method is called, and it triggers the check_empty_entries() method to ensure that there are no empty texts or rationales. If any empty entries are found, the method raises a ValueError and halts further processing.
# - __getitem__ function: 
#     - Returns a single data sample (text and its corresponding rationale) based on the index (idx).
#     - If the index is out of range or if any empty text/rationale is encountered, it raises an appropriate error.
#     - **Tokenization**:
#         - The text is tokenized using the provided BERT tokenizer, which converts each word into one or more subword tokens.
#         - The tokenizer.encode() function is used to convert the tokens into BERT input IDs, with special tokens like [CLS] and [SEP] added.
#     - **Rationale Alignment with Tokens**: Since a word can be split into multiple subwords by the BERT tokenizer, the rationale (which applies to full words) needs to be expanded to cover each subword. This loop ensures that if a word has a rationale of 1 (important), all its subwords will also be labeled as 1.
#     - Adjusting for Special Tokens: The rationale_extended list is adjusted by adding 0s at the start and end to account for the special [CLS] and [SEP] tokens, which are not part of the original text.
#     - **Masking Important Tokens**:
#         - Identifying Important Tokens: Tokens marked as important (where rationale_extended has a value of 1) are identified as mask_positions.
#         - Masking Based on Percentage: A percentage (defined by self.mask_percentage) of the important tokens are randomly selected for masking. These tokens are replaced with the BERT [MASK] token in input_ids.
#         - In typical BERT training, the standard **percentage of tokens to be masked is 15%**, meaning only 15% of the tokens in the input sequence are randomly masked for the MLM objective.
#     - **Padding**: The input sequence is padded to the max_len with BERT’s padding token ID. For the labels (which are used for loss computation), the padding tokens are assigned a value of -100, which tells the loss function to ignore these positions.

# In[24]:


class TextDataset(Dataset):
    def __init__(self, texts, rationales, tokenizer, max_len=val_max_len, mask_percentage=0.15):
        self.tokenizer = tokenizer
        self.texts = texts
        self.rationales = rationales
        self.max_len = max_len
        self.mask_percentage = mask_percentage
        self.check_empty_entries() # H!split Check for empty texts or rationales during initialization
        
    def check_empty_entries(self): # H!split
        # Check if any text or rationale is empty and log their indices
        empty_text_indices = [i for i, text in enumerate(self.texts) if not text]
        empty_rationale_indices = [i for i, rationale in enumerate(self.rationales) if not rationale]

        if empty_text_indices:
            print(f"Empty text entries found at indices: {empty_text_indices}")
        if empty_rationale_indices:
            print(f"Empty rationale entries found at indices: {empty_rationale_indices}")

        # Raise an error if there are any empty entries
        if empty_text_indices or empty_rationale_indices:
            raise ValueError("Empty texts or rationales found. Please clean the dataset.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx >= len(self.texts) or idx >= len(self.rationales):
            raise IndexError("Index out of range for 'texts' or 'rationales'.")
        
        text = self.texts[idx]
        rationale = self.rationales[idx]
        
        if not text or not rationale:  # H!split Additional check for empty inputs
            raise ValueError(f"Empty text or rationale found at index {idx}")

        # Tokenization
        tokens = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.encode(tokens, add_special_tokens=True)

        # Apply rationale to each subword
        rationale_extended = []
        for word, rat in zip(text.split(), rationale):
            # Tokenize the word into subwords
            subwords = self.tokenizer.tokenize(word)
            # Create a list of rationale labels for each subword
            subword_labels = [rat] * len(subwords)
            # Add the list of rationale labels to the extended rationale list
            rationale_extended.extend(subword_labels)
        rationale_extended = [0] + rationale_extended + [0]  # Adjust for [CLS] and [SEP]

        # Masking
        mask_positions = [i for i, r in enumerate(rationale_extended) if r == 1]
        # Determine the number of positions to mask based on the mask percentage.
        # Calculate the number of masks to apply by multiplying the total number of
        # Maskable positions by the mask percentage and converting it to an integer.
        num_to_mask = int(len(mask_positions) * self.mask_percentage)
        # Randomly select 'num_to_mask' positions from the list of maskable positions.
        # Ensure that each position is selected only once (no replacement).
        selected_masks = np.random.choice(mask_positions, num_to_mask, replace=False)
        labels = input_ids[:]  # Copy input_ids to labels
        for i in selected_masks:
            input_ids[i] = self.tokenizer.mask_token_id # Replace selected tokens with mask token

        # Padding
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length) #creates a list of padding token IDs with a length equal to padding_length.
            labels.extend([-100] * padding_length)  # -100 for padding in loss computation

        return torch.tensor(input_ids), torch.tensor(labels)


# - Loading BERT Components:
#     - The BertTokenizer is loaded using the pre-trained bert-base-uncased model, which handles tokenization.
#     - The BertForMaskedLM model is loaded for performing the Masked Language Modeling task.
# - Device Configuration: The model is transferred to the appropriate device (GPU if available, otherwise CPU).

# In[25]:


# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[26]:


# texts = input_data


# In[27]:


# Example labels and input data
labels = list(df['label'])
input_data = list(df['input_text'])

# Label Encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


# Printing to observe how they are encoded:

# In[28]:


# Print the original labels and their corresponding encoded values
print("Original Labels:", labels[:10])  # Print first 10 original labels as a sample
print("Encoded Labels:", encoded_labels[:10])  # Print first 10 encoded labels as a sample

# Print the mapping of the label encoder (i.e., which label maps to which number)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nLabel Mapping (Original Label -> Encoded Value):")
for original_label, encoded_value in label_mapping.items():
    print(f"{original_label}: {encoded_value}")


# This code performs data cleaning by removing any entries where the input text or rationales are empty. 
# - Even though the check_empty_entries method checks for empty entries inside the TextDataset class, you may still need the manual cleaning part of the code.
# - This ensures that your data is already clean and avoids triggering the error in check_empty_entries during dataset initialization.
# 
# After cleaning, the data is split into training and validation sets for model training. 
# 
# Additionally, it prints the number of cleaned (removed) entries and the size of the training and validation sets.
# 
# Train-test split:
# - test_size=0.2: Common and widely used choice (0.8 Train - 0.2 Test)
# - random_state=42: A random seed that ensures reproducibility of the split. Using the same random_state value will give you the same split every time you run the code.

# In[188]:


# Initialize counters for cleaned entries
num_cleaned = 0

# Lists to store cleaned data
cleaned_input_data = []
cleaned_rationales = []
cleaned_labels = []

# Iterate over input_data, rationales, and labels
for i in range(len(input_data)):
    # Check if both the input text and rationale are non-empty
    if input_data[i] and rationales[i]:
        cleaned_input_data.append(input_data[i])
        cleaned_rationales.append(rationales[i])
        cleaned_labels.append(encoded_labels[i])
    else:
        # Increment the counter for cleaned entries
        num_cleaned += 1

# Now use these cleaned versions of the data
input_data = cleaned_input_data
rationales = cleaned_rationales
encoded_labels = cleaned_labels

# Print how many entries were cleaned
print(f"Number of entries cleaned: {num_cleaned}")

# Train-test split
train_texts, val_texts, train_labels, val_labels, train_rationales, val_rationales = train_test_split(
    input_data, encoded_labels, rationales, stratify=encoded_labels, test_size=0.2, random_state=42
)

print(f"Training set size: {len(train_texts)}")
print(f"Validation set size: {len(val_texts)}")
print(f"Training rationales size: {len(rationales[:len(train_texts)])}")
print(f"Validation rationales size: {len(rationales[len(train_texts):])}")


# This code defines a function called evaluate_1st_model that evaluates a BERT-based model on a dataset, measuring its accuracy. 
# 
# The model is evaluated in a batch-wise manner using a data loader, and special tokens like [CLS], [SEP], and padding tokens are excluded from the accuracy calculation.
# 
# - Generate attention mask: 1 for real tokens, 0 for padding tokens
# - torch.no_grad(): Disables gradient calculation during evaluation to save memory and computation, since gradients are not needed for backpropagation during evaluation.
# - Differentiate between mask and attention_mask:
#     - attention_mask ensures the model processes the input correctly, only focusing on real data and ignoring padded tokens.
#     - mask ensures that accuracy is only calculated for meaningful tokens (excluding padding and special tokens like [CLS] and [SEP]), giving a more accurate measure of the model's performance.
#     - The reason we don't remove special tokens like [CLS] and [SEP] when creating the attention mask is that these tokens play an important role in BERT’s architecture during the forward pass and must be attended to by the model.

# In[189]:


# H!1st
def evaluate_1st_model(model, dataloader, tokenizer, device):
    model.eval()
    total_accuracy = 0
    num_batches = 0

    for batch in dataloader:
        inputs, labels = batch[0].to(device), batch[1].to(device)

        # Generate attention mask: 1 for real tokens, 0 for padding tokens
        attention_mask = (inputs != tokenizer.pad_token_id).long().to(device)

        # Disable gradient calculation
        with torch.no_grad():
            # Model Forward Pass: Get the model outputs (logits) for the input data
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Create a mask to exclude special tokens ([CLS], [SEP], [PAD]) from accuracy calculation        
            mask = (labels != -100) & (inputs != tokenizer.cls_token_id) & (inputs != tokenizer.sep_token_id)
            
            # Calculate Accuracy for the Batch
            correct = (predictions == labels) & mask
            accuracy = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0

            total_accuracy += accuracy
            num_batches += 1

    average_accuracy = total_accuracy / num_batches

    return average_accuracy


# In[ ]:


# # Dataset and DataLoader
# dataset = TextDataset(texts, rationales, tokenizer, max_len=val_max_len, mask_percentage=mask_percentage)
# dataloader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset)) # H! batch_size used to be 16


# Two datasets are created, one for training (train_mlm_dataset) and one for validation (val_mlm_dataset), using the TextDataset class that prepares the data for the Masked Language Model (MLM) task.

# In[ ]:


# Create dataset for MLM
train_mlm_dataset = TextDataset(train_texts, rationales[:len(train_texts)], tokenizer, max_len=val_max_len, mask_percentage=mask_percentage)
val_mlm_dataset = TextDataset(val_texts, rationales[len(train_texts):], tokenizer, max_len=val_max_len, mask_percentage=mask_percentage)


# DataLoaders are created for both the training and validation datasets. DataLoaders handle batching and shuffling the data during training.
# 
# RandomSampler: Ensures that the dataset is shuffled randomly, so that each training epoch sees the data in a different order, improving generalization.

# In[ ]:


# Create DataLoaders for MLM training
train_mlm_loader = DataLoader(train_mlm_dataset, batch_size=batch_size, sampler=RandomSampler(train_mlm_dataset))
val_mlm_loader = DataLoader(val_mlm_dataset, batch_size=batch_size, sampler=RandomSampler(val_mlm_dataset))


# optimizer:
# - AdamW: This is the Adam with Weight Decay optimizer, commonly used in training transformer-based models like BERT.
# - model.parameters(): The parameters of the model that need to be updated during training are passed to the optimizer.
# - lr=2e-5: The learning rate is set to 2e-5. This is a common value for fine-tuning pre-trained models like BERT, as higher learning rates can cause instability.
# - weight_decay=0.01: This adds L2 regularization to the optimizer to help prevent overfitting. Weight decay penalizes large weights in the model, encouraging it to generalize better.
#     - Value 0.01 is a common and standard choice for training transformer-based models like BERT, particularly when using the AdamW optimizer.

# In[ ]:


# optimizer = AdamW(model.parameters(), lr=2e-5)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01) # H!regu


# The total number of training steps is calculated by:
# - num_epochs: The number of epochs (complete passes through the training data).
# - len(train_mlm_loader): The number of batches per epoch (determined by the size of the dataset and batch size).

# In[ ]:


# num_training_steps = num_epochs * len(dataloader)
num_training_steps = num_epochs * len(train_mlm_loader)


# - Learning Rate Scheduler:
#     - get_scheduler("linear"): This function creates a linear learning rate scheduler, which gradually reduces the learning rate from the initial value (lr=2e-5) to zero over the course of training.
#     - num_warmup_steps: There is no warm-up period, meaning the learning rate starts from its initial value and then linearly decays throughout training.
#         - Learning rate warm-up is a technique used to gradually increase the learning rate from a small value to its specified initial value over a certain number of steps at the beginning of training.
#     - num_training_steps: The total number of training steps over which the learning rate will decay.
# 
# 
# - Why Use a Learning Rate Scheduler?
#     - Improved Convergence: 
#         - During training, starting with a higher learning rate can help the model converge faster in the early stages, while lowering the learning rate later can help the model fine-tune its parameters without overshooting or diverging.
#         - A scheduler can gradually reduce the learning rate, allowing the model to make smaller updates in later stages, improving convergence and final performance.
#     - Avoiding Overshooting:
#         - A constant learning rate can cause the optimizer to take steps that are too large, especially later in training, leading to overshooting or oscillations around a good solution.
#         - A scheduler gradually reduces the learning rate to avoid this issue.
#     - Fine-tuning Pre-trained Models (like BERT): **Pre-trained models like BERT benefit from a smaller learning rate during fine-tuning** because the weights are already optimized for general tasks. A scheduler helps by starting with an appropriate learning rate and then gradually reducing it.
# 
# 
# - Warmup:
#     - Including a warmup period in both Masked Language Modeling (MLM) and Sequence Classification tasks can be beneficial, particularly when fine-tuning pre-trained models like BERT.
#     - The purpose of warmup is to prevent large initial weight updates, which can destabilize training, especially when starting with randomly initialized or pre-trained weights.
#     - Pre-trained weights may be sensitive to large changes early in training, and a gradual warmup prevents these drastic updates.

# In[ ]:


lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)


# Training Loop for Each Batch:
# - Unpacking the Batch: Each batch from train_mlm_loader contains tokenized input sequences (input_ids) and their corresponding labels. These are moved to the appropriate device (CPU or GPU).
# - Attention Mask: The attention mask is generated similar to evaluate_1st_model().
# - Zeroing Gradients: optimizer.zero_grad() clears the gradients from the previous step to ensure they don’t accumulate.
# - Forward Pass: The model processes the input, generating predictions and computing the loss.
# - Backward Pass: loss.backward() computes the gradients based on the loss. These gradients are then used to update the model's parameters via optimizer.step().
# - Learning Rate Scheduler: After each optimizer step, the learning rate scheduler (lr_scheduler) is updated to adjust the learning rate according to the linear schedule.
# 
# Validation After Each Epoch:
# - Evaluation Mode: model.eval() sets the model to evaluation mode, ensuring that layers like dropout are disabled during validation.
# - No Gradient Calculation: torch.no_grad() is used to disable gradient calculation during the validation phase, which saves memory and computation since gradients are not needed for evaluation.
# - Validation Loop: Similar to the training loop, but without the backward pass. The model's performance is evaluated on the validation set by calculating the total validation loss for the epoch.
# 
# Early Stopping Logic:
# - Check for Improvement: If the current validation loss improves by more than min_delta compared to best_val_loss, it is considered an improvement. In that case:
#     - The best_val_loss is updated.
#     - epochs_no_improve is reset to 0.
#     - The model's state is saved to 'best_model.pt' (the best version of the model so far).
# - No Improvement: If the validation loss does not improve, the epochs_no_improve counter is incremented.
# - Early Stopping: If the validation loss hasn’t improved for a number of consecutive epochs equal to patience (in this case, 3 epochs), training is stopped early with a message indicating that early stopping has been triggered.
# 
# After early stopping, the best version of the model (saved earlier) is loaded from the file 'best_model.pt'. This ensures that the model used after training is the one that performed best on the validation set.
# 
# Once training is complete, the fine-tuned model is saved to the directory './fine_tuned_bert'.

# In[190]:


if bool_1st_early_stop: # H!
    # Early stopping parameters
    patience = 3  # Number of epochs to wait for improvement
    min_delta = 0.001  # Minimum change to qualify as an improvement
    best_val_loss = float('inf')  # Initialize with a large value
    epochs_no_improve = 0

    # Training loop with early stopping
    for epoch in range(num_epochs):
        model.train() # sets the model to "training mode," which ensures that specific layers like dropout or batch normalization behave appropriately during training
        total_train_loss = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch in train_mlm_loader:
            # Unpack the batch to get input_ids and labels, and generate attention masks
            input_ids, labels = batch[0].to(device), batch[1].to(device)

            # Create the attention mask: 1 for real tokens, 0 for padding tokens
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            # Zero out the gradients
#             model.zero_grad() # H!split
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_train_loss += loss.item()

        # Evaluate the model after each epoch
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_mlm_loader:
                input_ids, labels = batch[0].to(device), batch[1].to(device)
                attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = outputs.loss
                total_val_loss += val_loss.item()
        print(f"Epoch {epoch + 1}: Train Loss = {total_train_loss}, Val Loss = {total_val_loss}")
        
        # Compute average accuracy
        average_accuracy = evaluate_1st_model(model, val_mlm_loader, tokenizer, device)
        print(f"Epoch {epoch + 1}: Average Accuracy = {average_accuracy}")
        
        # Early stopping logic based on validation loss
        if best_val_loss - total_val_loss > min_delta:  # Improvement threshold
            best_val_loss = total_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')  # Save the best model
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model after early stopping
    model.load_state_dict(torch.load('best_model.pt', weights_only = True))

    # Save the fine-tuned model
    model.save_pretrained('./fine_tuned_bert')


# In[191]:


# if not bool_1st_early_stop:
#     # Training loop
#     model.train()
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch + 1}/{num_epochs}")
#         for batch in dataloader:
#             # H! add attention mask
#             # Unpack the batch to get input_ids and labels, and generate attention masks
#             input_ids, labels = batch[0].to(device), batch[1].to(device)

#             # Create the attention mask: 1 for real tokens, 0 for padding tokens
#             attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

#             # Zero out the gradients
#             model.zero_grad()

#             # Forward pass
#             # outputs = model(input_ids, labels=labels)
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels) # H! add attention mask
#             loss = outputs.loss

#             # Backward pass
#             loss.backward()
#             optimizer.step()
#             lr_scheduler.step()

#             # Optionally, print the loss for each batch (uncomment if needed)
# #             print(f"Batch Loss: {loss.item()}")

#         # H!1st Evaluate the model after each epoch
#         evaluate_1st_model(model, dataloader, tokenizer, device)

#     # Save the fine-tuned model
#     model.save_pretrained('./fine_tuned_bert')


# In[192]:


# Zip to download
if bool_zip:
    get_ipython().system('zip -r bert_mlm_model.zip /kaggle/working/fine_tuned_bert')


# # Fine-tuning 2: Supervised Sequence Classification Fine-Tuning
# 
# **Objective:**
# After MLM fine-tuning, the model is further fine-tuned for sequence classification. In this step, the model is trained to classify entire sequences (e.g., sentences or posts) into specific categories (e.g., 'normal', 'hatespeech').
# 
# **Purpose:**
# This step adapts the MLM-fine-tuned BERT model to the specific classification task, leveraging the understanding it gained from MLM to make accurate predictions based on labeled data.
# 
# **Implementation:**
# - The preprocessed data is split into training and validation sets.
# - The BERT model is loaded (from the MLM fine-tuned version) and further fine-tuned on the classification task using the train_texts and train_labels.
# - The model is trained using a supervised learning approach with cross-entropy loss, optimizing it to correctly classify the input sequences based on the given labels.

# After MLM fine-tuning previously, the MLM model is saved. Here we load this fine-tuned model and further fine-tune it for the sequence classification task.

# In[195]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
# Use pretrained model from the first step
model = BertForSequenceClassification.from_pretrained('/kaggle/working/fine_tuned_bert', num_labels=len(np.unique(train_labels))) 


# This code defines a function called tokenize_and_format which handles tokenization and formatting of text data to be used in a PyTorch dataset. 
# 
# It converts the input text into the format expected by a BERT-like model (i.e., token IDs, attention masks) and packages them together with the labels into a TensorDataset.
# - TensorDataset: This is a PyTorch utility that bundles together multiple tensors (in this case, input_ids, attention_mask, and labels) so they can be used together in a DataLoader for batching.

# In[196]:


# Tokenization and input formatting
def tokenize_and_format(texts, labels):
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=val_max_len, return_tensors="pt")
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))
    return dataset


# The model is trained on the training dataset (train_dataset) and validated on the validation dataset (val_dataset). This step adapts the model specifically to the task of classifying sequences based on the labeled data.

# In[197]:


# train_dataset = tokenize_and_format(train_texts, train_labels)
# val_dataset = tokenize_and_format(val_texts, val_labels)

# Tokenization and input formatting for sequence classification
train_seq_dataset = tokenize_and_format(train_texts, train_labels)
val_seq_dataset = tokenize_and_format(val_texts, val_labels)


# In[198]:


# DataLoader setup
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
train_seq_loader = DataLoader(train_seq_dataset, batch_size=batch_size, shuffle=True)
val_seq_loader = DataLoader(val_seq_dataset, batch_size=batch_size, shuffle=False)


# In[199]:


# Move model to GPU if available
device = torch.device("cuda")
model.to(device)


# In[200]:


# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.01) # H!regu
# optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


# - Initializes the CrossEntropyLoss function in PyTorch, which is commonly used for classification tasks.
# - Both Sequence Classification and Masked Language Modeling (MLM) tasks involve using cross-entropy loss, but the way it's applied differs between them.
#     - Sequence classification: Explicitly define and apply the CrossEntropyLoss to compare these logits with the true class labels.
#         - Can also be handled internally.
#     - MLM:  When using libraries like Hugging Face's Transformers, providing labels to the model automatically computes the loss internally.
#         - loss = outputs.loss  # Cross-entropy loss computed internally
# - Understanding the Two Tasks:
#     - a. Sequence Classification
#         - Objective: Assign a single label or multiple labels to an **entire sequence of text** (e.g., sentiment analysis, topic classification).
#         - Output: A fixed-size vector representing the logits (unnormalized probabilities) for each class.
#         - Loss Calculation: Compare the predicted logits with the true class labels using a loss function like CrossEntropyLoss.
#     - b. Masked Language Modeling (MLM)
#         - Objective: **Predict missing (masked) tokens in a sequence**, effectively learning **contextual** representations.
#         - Output: Predictions for each token in the sequence, but only the masked tokens contribute to the loss.
#         - Loss Calculation: Compute the loss between the predicted token probabilities and the actual masked tokens using a loss function like CrossEntropyLoss.

# In[ ]:


num_training_steps = num_epochs * len(train_seq_loader)
lr_scheduler = get_scheduler(
    "linear",  # You can use "linear" or another type of scheduler
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps,
)


# In[ ]:


# loss_fn = torch.nn.CrossEntropyLoss()


# Implicit cross entropy loss calculation: loss = outputs.loss
# 
# Training similar to MLM.

# In[201]:


# Training loop function definition
def train(epoch):
    model.train()
    # Initialize a variable to keep track of the total loss across all batches in this epoch.
    total_loss = 0
    # Loop over each batch in the training data loader. The DataLoader object handles batching for us.
    for batch in train_seq_loader:
        # Convert the batch items into a tuple and move each item to the appropriate device (GPU or CPU).
        # The 'batch' is a tuple containing (input_ids, attention_mask, labels).
        batch = tuple(item for item in batch)
        # Create a dictionary to hold the model inputs. These inputs are moved to the appropriate device (GPU/CPU).
        inputs = {
            'input_ids': batch[0].to(device),          # Tensor of token IDs
            'attention_mask': batch[1].to(device),     # Tensor of attention masks (1 for real tokens, 0 for padding tokens)
            'labels': batch[2].to(device)              # Tensor of labels (used for calculating loss)
        }
        # Reset gradients for the optimizer. This is necessary because gradients accumulate by default in PyTorch.
        optimizer.zero_grad()
        # Forward pass: the model processes the inputs and returns the outputs. The outputs include the loss value.
        outputs = model(**inputs)
        # Extract the loss value from the model's outputs.
        loss = outputs.loss
        # Backward pass: compute the gradient of the loss with respect to model parameters.
        loss.backward()
        # Update the model's parameters using the gradients computed in the backward pass.
        optimizer.step() 
        lr_scheduler.step() # H!
        total_loss += loss.item()
    average_loss = total_loss / len(train_seq_loader)
    print(f'Epoch {epoch}, Training loss: {average_loss}')


# Simplified training for quick testing

# In[202]:


# H!
def quick_train(epoch):
    model.train()
    total_loss = 0
    
    # Limit the number of batches for quick testing
    max_batches = 5
    
    for i, batch in enumerate(train_seq_loader):
        if i >= max_batches:  # Only process a limited number of batches
            break

        # Move batch items to the appropriate device (GPU/CPU)
        batch = tuple(item.to(device) for item in batch)
        inputs = {
            'input_ids': batch[0],          # Tensor of token IDs
            'attention_mask': batch[1],     # Tensor of attention masks (1 for real tokens, 0 for padding tokens)
            'labels': batch[2]              # Tensor of labels (used for calculating loss)
        }
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Calculate the average loss across processed batches
    average_loss = total_loss / (i + 1)
    print(f'Epoch {epoch}, Training loss: {average_loss}')


# After training, the model is evaluated using metrics like AUROC (Area Under the Receiver Operating Characteristic curve) and classification reports.

# In[203]:


def evaluate(loader):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = tuple(item.to(device) for item in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            labels = batch[2]
            outputs = model(**inputs)
            logits = outputs.logits
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # H! Convert lists of numpy arrays to single numpy arrays
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate probabilities
    probabilities = F.softmax(torch.tensor(all_logits), dim=1).numpy()

    # Calculate AUROC for each class (assuming binary or multi-class classification)
    if probabilities.shape[1] == 2:  # Binary classification
        auroc = roc_auc_score(all_labels, probabilities[:, 1])
    else:  # Multi-class classification
        auroc = roc_auc_score(all_labels, probabilities, multi_class="ovr")
    print(f"AUROC: {auroc}")

    # Print the classification report with 3 decimal places
    report = classification_report(all_labels, np.argmax(probabilities, axis=1), target_names=label_encoder.classes_, digits=3)
    print(report)


# In[204]:


# H!regu
if bool_early_stop:
    # Parameters for Early Stopping
    patience = 3  # Number of epochs with no improvement after which training will be stopped
    min_delta = 0.001  # Minimum change to qualify as an improvement
    best_val_loss = float('inf')  # Initialize with a very large value
    epochs_no_improve = 0  # Counter for how many epochs have passed without improvement
    early_stop = False  # Flag to indicate if training should stop

    # Train the model with Early Stopping
    for epoch in range(1, num_epochs_2nd + 1):
#         model.train()
#         total_train_loss = 0

        # Training step
        if fast_run:
            quick_train(epoch)
        else:
            train(epoch)

        # Validation step
        if bool_evaluate:
#             model.eval()
#             val_loss = 0
#             with torch.no_grad():
#                 for batch in train_seq_loader:
#                     input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

#                     optimizer.zero_grad()
#                     outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#                     loss = outputs.loss
#                     loss.backward()
#                     optimizer.step()
#                     total_train_loss += loss.item()

#             val_loss /= len(val_loader)
#             print(f'Epoch {epoch}, Validation loss: {val_loss}')

            # Evaluate on validation set after each epoch
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_seq_loader:
                    input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss = outputs.loss
                    total_val_loss += val_loss.item()
            print(f"Epoch {epoch}: Train Loss = {total_train_loss}, Val Loss = {total_val_loss}")
            
            if bool_evaluate:
                evaluate(val_seq_loader)

            # Early Stopping Logic with min_delta
            if total_val_loss < best_val_loss - min_delta:
                best_val_loss = total_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), 'best_model.pt')  # Save best model
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break
#             if best_val_loss - val_loss > min_delta:  # Significant improvement
#                 best_val_loss = val_loss
#                 epochs_no_improve = 0
#                 torch.save(model.state_dict(), 'best_model.pt')  # Save the best model
#             else:
#                 epochs_no_improve += 1

#             if epochs_no_improve >= patience:
#                 print("Early stopping triggered.")
#                 early_stop = True
#                 break

#         # Check for early stopping
#         if early_stop:
#             break

    # Load the best model after early stopping
    model.load_state_dict(torch.load('best_model.pt', weights_only=True))


# In[205]:


# # Train the model
# if not bool_early_stop:
#     for epoch in range(num_epochs_2nd):
#         epoch += 1
#         print(epoch)
#         if not fast_run:
#             train(epoch)
#         else:
#             quick_train(epoch)
#         if bool_evaluate:
#             evaluate(val_loader)


# # Save

# In[206]:


# H!
if bool_zip:
    # Define the directory where both model and tokenizer will be saved
    output_directory = './final_fine_tuned_bert_2_class'

    # Save the model
    model.save_pretrained(output_directory)

    # Save the tokenizer
    tokenizer.save_pretrained(output_directory)

    # H! zip
    get_ipython().system('zip -r bert_cf_model.zip final_fine_tuned_bert_2_class/')


# In[207]:


from sklearn.metrics import roc_auc_score, classification_report
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer


# # Model Evaluation and LIME Explanations

# In[208]:


# import numpy as np
# from sklearn.metrics import roc_auc_score, classification_report
# from lime.lime_text import LimeTextExplainer
# from transformers import BertTokenizer
# import torch

# # H!
# def evaluate(loader, model, label_encoder, tokenizer, device):
#     model.eval() # The model is set to evaluation mode to disable dropout and other training-specific behavior.
#     all_logits = []
#     all_labels = []
#     explainer = LimeTextExplainer(class_names=label_encoder.classes_)
# #     logits_shapes = [] # H!

#     with torch.no_grad():
#         for i, batch in enumerate(loader):
#             # H! Skip the batch if it's empty
#             if batch[0].size(0) == 0:
#                 continue
            
#             batch = tuple(item.to(device) for item in batch)
#             inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
#             labels = batch[2]
#             outputs = model(**inputs)
#             logits = outputs.logits
            
# #             all_logits.append(logits.cpu().numpy())
# #             all_labels.extend(labels.cpu().numpy())
#             # H! Only append if logits and labels are non-empty
#             if logits.size(0) > 0:
#                 all_logits.append(logits.cpu().numpy())
# #                 logits_shapes.append(logits.cpu().numpy().shape)
#             if labels.size(0) > 0:
#                 all_labels.append(labels.cpu().numpy())

#             # Using LIME to explain the first instance in the batch
#             if i == 0 and len(batch[0]) > 0:  # Check if batch is non-empty
#                 text_instance = tokenizer.decode(batch[0][0], skip_special_tokens=True)
                
#                 # Define the prediction function for LIME
#                 def predict(texts):
#                     # Convert text to model inputs
#                     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
#                     inputs = {key: value.to(device) for key, value in inputs.items()}
#                     with torch.no_grad():
#                         logits = model(**inputs).logits
#                     # Return probabilities
#                     return F.softmax(logits, dim=1).cpu().numpy()

#                 exp = explainer.explain_instance(
#                     text_instance, 
#                     predict,
#                     num_features=6, 
#                     labels=[labels[0].item()]
#                 )
#                 exp.show_in_notebook(text=True)

#     # H! Debugging: Check the contents of logits and labels
# #     print(f"Logits shapes: {logits_shapes}")
# #     print(f"Type of elements in all_labels: {type(all_labels[0])}")
# #     print(f"Sample values from all_labels: {all_labels[:10]}")
                
#     # H! Concatenate all logits into a single numpy array before converting to tensor
#     if len(all_logits) > 0 and len(all_labels) > 0:
#         all_logits = np.concatenate(all_logits, axis=0)
#         all_labels = np.concatenate(all_labels, axis=0)
#     else:
#         print("Warning: No data to evaluate.")
#         return

#     # Calculate probabilities
#     probabilities = F.softmax(torch.tensor(all_logits), dim=1).numpy()

#     # Calculate AUROC for each class
#     if probabilities.shape[1] == 2:  # Binary classification
#         auroc = roc_auc_score(all_labels, probabilities[:, 1])
#     else:  # Multi-class classification
#         auroc = roc_auc_score(all_labels, probabilities, multi_class="ovr")
#     print(f"AUROC: {auroc}")

#     # Print the classification report
#     report = classification_report(all_labels, np.argmax(probabilities, axis=1), target_names=label_encoder.classes_, digits=3)
#     print(report)

# # Call evaluate function
# evaluate(val_loader, model, label_encoder, tokenizer, device)


# In[209]:


# from lime.lime_text import LimeTextExplainer

# # H!
# def explain_prediction(text, model, tokenizer, df, label_encoder, device='cuda'):
#     """Generate explanation for a given text using LIME and include the target of hate speech if applicable."""
#     model.eval()  # Set the model to evaluation mode
#     explainer = LimeTextExplainer(class_names=label_encoder.classes_)

#     # Define the prediction function for LIME
#     def model_predict(texts):
#         # Convert texts to input format for the model
#         inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
#         inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU
#         with torch.no_grad():
#             logits = model(**inputs).logits
#         # Return softmax probabilities
#         return F.softmax(logits, dim=1).cpu().numpy()

#     # Generate explanation using LIME
#     exp = explainer.explain_instance(
#         text, 
#         model_predict,  # Prediction function
#         num_features=6,  # Number of features to show
#         top_labels=1     # Explain the top label predicted by the model
#     )

#     # Get the predicted label
#     predicted_label = exp.top_labels[0]
#     predicted_label_name = label_encoder.inverse_transform([predicted_label])[0]

#     # Display the prediction result
#     print(f"Predicted Label: {predicted_label_name}")

#     # Only find the target group if the predicted label is "hatespeech"
#     if predicted_label_name == 'hatespeech':
#         # Normalize the text for matching
#         text_normalized = text.lower().strip()

#         # Normalize the DataFrame input_text for matching
#         df['input_text_normalized'] = df['input_text'].str.lower().str.strip()

#         # Attempt to find the closest matching text in the DataFrame
#         matching_row = df[df['input_text_normalized'] == text_normalized]

#         if matching_row.empty:
#             print(f"No matching text found in the DataFrame for the input: '{text}'")
#             target_group = "N/A"
#         else:
#             target_group = matching_row['Final_target'].values[0]

#         print(f"Target of Hate Speech: {target_group}")

#     # Show the LIME explanation
#     exp.show_in_notebook(text=True)


# In[210]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np


# In[211]:


pip install sentence-transformers


# In[212]:


from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained Sentence-BERT model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def explain_prediction(text, model, tokenizer, df, label_encoder, device='cuda'):
    """Generate explanation for a given text using LIME and include the target of hate speech if applicable."""
    model.eval()  # Set the model to evaluation mode
    explainer = LimeTextExplainer(class_names=label_encoder.classes_)

    # Define the prediction function for LIME
    def model_predict(texts):
        # Convert texts to input format for the model
        inputs = tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512,
#             clean_up_tokenization_spaces=True,  # Recommended to match current behavior
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU
        with torch.no_grad():
            logits = model(**inputs).logits
        # Return softmax probabilities
        return F.softmax(logits, dim=1).cpu().numpy()

    # Generate explanation using LIME
    exp = explainer.explain_instance(
        text, 
        model_predict,  # Prediction function
        num_features=6,  # Number of features to show
        top_labels=1     # Explain the top label predicted by the model
    )

    # Get the predicted label
    predicted_label = exp.top_labels[0]
    predicted_label_name = label_encoder.inverse_transform([predicted_label])[0]

    # Display the prediction result
    print(f"Predicted Label: {predicted_label_name}")

    # Only find the target group if the predicted label is "hatespeech"
    if predicted_label_name != 'normal':
        # Compute embeddings for the input text and all texts in the dataset
        input_embedding = embedding_model.encode(text, convert_to_tensor=True)
        dataset_embeddings = embedding_model.encode(df['input_text'].tolist(), convert_to_tensor=True)

        # Calculate cosine similarity between the input text and all dataset texts
        cosine_similarities = cosine_similarity(input_embedding.cpu().numpy().reshape(1, -1), 
                                                dataset_embeddings.cpu().numpy())[0]

        # Find the index of the most similar text in the dataset
        most_similar_index = np.argmax(cosine_similarities)
        
        # Get the target group for the most similar text
        target_group = df.iloc[most_similar_index]['Final_target']
        print(f"Target of Hate Speech: {target_group}")

    # Show the LIME explanation
    exp.show_in_notebook(text=True)


# In[213]:


# Example usage
text = "That bitch doesnt know what to do with that ugly face"
# H! added df and label_encoder
explain_prediction(text, model, tokenizer, df, label_encoder)


# In[214]:


# Example usage
text = "nawt yall niggers ignoring me"
# H! added df and label_encoder
explain_prediction(text, model, tokenizer, df, label_encoder)

