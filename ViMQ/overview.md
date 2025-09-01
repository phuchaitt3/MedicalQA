An Exploratory Data Analysis (EDA) of the ViMQ dataset, a Vietnamese Medical Question Dataset, involves examining its structure and content to uncover patterns, and characteristics. This analysis is crucial for understanding the data's nuances before any machine learning modeling.

Understanding the ViMQ Dataset Files

The ViMQ dataset is designed for Natural Language Processing (NLP) tasks, specifically for <span style="background:#9254de">building healthcare dialogue systems</span>. It focuses on <span style="background:#9254de">two main tasks: Named Entity Recognition (NER) and Intent Classification</span>. The files you've provided each serve a distinct purpose within this dataset:

*   **`train.json`, `dev.json`, and `test.json`**: These are the core data files, containing the annotated medical questions. They are split into three sets:
    *   `train.json`: Used to train a machine learning model. It typically contains the largest number of data samples.
    *   `dev.json` (development set): Used to tune the model's hyperparameters and for validation during the training process. It is also known as the validation set.
    *   `test.json`: Used to evaluate the final performance of the trained model on unseen data.
    Each of these JSON files contains a list of entries, where each entry represents a medical question and includes the sentence, its sequence labels (for NER), and its sentence-level label (for intent classification).

```json
{
    "sentence": "Có nên dùng vitamin b12 sau khi phẫu_thuật ung_thư dạ_dày không ?",
    "seq_label": [
        [
            3,
            4,
            "drug"
        ],
        [
            7,
            7,
            "medical_procedure"
        ],
        [
            8,
            9,
            "SYMPTOM_AND_DISEASE"
        ]
    ],
    "sent_label": "method_diagnosis"
}
```
[<1>]

*   **`entity_set.txt`**: This file defines the <span style="background:#9254de">set of all possible entity labels used in the Named Entity Recognition task</span>. For the ViMQ dataset, these entities are related to the medical domain, such as "SYMPTOM_AND_DISEASE," "medical_procedure," and "drug."
```
UNK
O
SYMPTOM_AND_DISEASE
medical_procedure
drug
```

*   **`char2index.json`**: This file is a vocabulary mapping where each unique character in the dataset is assigned a specific integer index. This is a <span style="background:#9254de">common preprocessing step for feeding text data into certain types of neural network models</span>.

---
**Files to Target for EDA**

For an EDA survey, the primary files to target are **`train.json`**, **`dev.json`**, and **`test.json`**. These files contain the actual text data and its corresponding annotations, which are the main subjects of interest for EDA. The `entity_set.txt` and `char2index.json` files provide context about the labels and character vocabulary, which will be useful during the analysis of the core data files.

Steps for Performing EDA on the ViMQ Dataset

Here is a structured approach to performing an EDA on the ViMQ dataset:

**1. Basic Data Exploration:**

*   **Dataset Size**: Count the number of samples in each of the `train.json`, `dev.json`, and `test.json` files to understand the data distribution across the splits.
*   **Sentence Analysis**:
    *   Calculate the length of each sentence (in terms of characters and words) to understand the distribution of question lengths.
    *   Visualize this distribution using histograms or box plots to identify any unusually long or short sentences.
    *   Analyze the vocabulary size by counting the number of unique words across the entire dataset.

**2. Intent Classification Analysis (`sent_label`):**

*   **Label Distribution**: Count the occurrences of each sentence-level label (e.g., "severity," "method_diagnosis," "treatment," "cause") in each data split.
*   **Visualization**: Create bar charts to visualize the distribution of these intent labels. This will reveal if the dataset is balanced or if certain intents are more frequent than others.

**3. Named Entity Recognition Analysis (`seq_label`):**

*   **Entity Frequency**: Count the occurrences of each entity type (from `entity_set.txt`) across all sentences in the dataset.
*   **Entity Distribution**: Visualize the frequency of each entity type using a bar chart to see which entities are most common (e.g., "SYMPTOM_AND_DISEASE" is likely to be frequent in a medical dataset).
*   **Entities per Sentence**: Calculate and visualize the number of entities present in each sentence. This can provide insights into the complexity of the medical questions.
*   **Entity Length**: Analyze the length (in words) of the text corresponding to each entity.

**4. Bivariate and Multivariate Analysis:**

*   **Intent vs. Entities**: Investigate if there is a correlation between the sentence-level intent and the types of entities present in the sentence. For example, do sentences with the intent "treatment" frequently contain the "drug" or "medical_procedure" entities?
*   **Word Clouds**: Generate word clouds for the text within each intent category and for each entity type to visually identify the most prominent terms.

**5. Text Preprocessing and Cleaning Analysis:**

*   **Character Set**: Examine the `char2index.json` file to understand the full range of characters present in the dataset, including special characters and accented Vietnamese characters.
*   **Tokenization**: Analyze how sentences are tokenized (split into words) and how this might affect NER, especially with Vietnamese-specific linguistic features.

By following these steps, you will gain a comprehensive understanding of the ViMQ dataset's structure, content, and characteristics, which is essential for any subsequent data preprocessing and model-building tasks.

# [1] Explanation of the entry
*   **`"sentence"`**:
    *   This is the medical question itself: "Có nên dùng vitamin b12 sau khi phẫu_thuật ung_thư dạ_dày không ?"
    *   *Translation:* "Should vitamin b12 be used after stomach cancer surgery?"

*   **`"seq_label"` (Sequence Labels for Named Entity Recognition - NER)**:
    *   This is a list of annotated entities found within the sentence. Each inner list `[start_token_index, end_token_index, entity_type]` indicates:
        *   `start_token_index`: The <span style="background:#9254de">starting index</span> of the token in the sentence.
        *   `end_token_index`: The <span style="background:#9254de">ending index</span> of the token in the sentence.
        *   `entity_type`: The category of the entity, as defined in `entity_set.txt`.
    *   For this example:
        *   `[3, 4, "drug"]`: The tokens at indices 3 and 4, "vitamin b12", are labeled as a "drug".
        *   `[7, 7, "medical_procedure"]`: The token at index 7, "phẫu_thuật" (surgery), is labeled as a "medical_procedure".
        *   `[8, 9, "SYMPTOM_AND_DISEASE"]`: The tokens at indices 8 and 9, "ung_thư dạ_dày" (stomach cancer), are labeled as a "SYMPTOM_AND_DISEASE".

*   **`"sent_label"` (Sentence-level Label for Intent Classification)**:
    *   This is the overall intent or category of the medical question.
    *   In this case, <span style="background:#9254de">`"method_diagnosis"`</span> indicates that the question is related to diagnosing a method or asking for a method of diagnosis/treatment.
