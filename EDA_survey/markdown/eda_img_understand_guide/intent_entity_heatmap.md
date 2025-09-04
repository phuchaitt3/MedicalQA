![[intent_entity_heatmap.png]]
General Rules for Understanding This Heatmap

This "co-occurrence heatmap" is a powerful visualization that shows the relationship between two categorical variables: the **intent of a sentence** and the **types of entities** found within it.

*   **Y-axis (Intent Label):** This represents the overall purpose of the question (e.g., asking about the `cause` of a condition).
*   **X-axis (Entity Type):** This represents the category of the specific medical term found in the question (e.g., `drug`).
*   **The Cells (Numbers and Colors):** Each cell shows a count. The number indicates how many times a specific entity type (column) appeared in a sentence with a specific intent (row). The color provides a quick visual cue:
    *   **Darker colors (hot spots):** Indicate a high frequency of co-occurrence. This suggests a strong relationship between that intent and entity type.
    *   **Lighter colors (cold spots):** Indicate a low frequency, meaning that combination is rare.
*   **Special "O" Column:** The `O` stands for "Outside." This column counts the number of sentences for each intent that have **zero** of the predefined medical entities.

---

Specific Observations and Interpretation for the ViMQ Heatmap

By analyzing the numbers and colors in your specific heatmap, we can extract several key insights about the nature of the medical questions in your dataset.

**Observation 1: The `SYMPTOM_AND_DISEASE` entity is the cornerstone of the entire dataset.**

*   **Evidence:** <span style="background:rgba(205, 244, 105, 0.55)">The `SYMPTOM_AND_DISEASE` column is overwhelmingly the "hottest" (darkest colors and highest numbers) for **every single intent category**.</span>
    *   `method_diagnosis`: 5965
    *   `treatment`: 3277
    *   `severity`: 1724
    *   `cause`: 1021
*   **Interpretation:** This is the most significant finding. It means that regardless of what users are asking, their questions are almost always anchored to a specific symptom or disease. This makes perfect sense intuitively: you ask for a *treatment for a headache*, the *cause of a fever*, or the *severity of cancer*.

**Observation 2: <span style="background:#9254de">The most common user behavior is asking for a diagnosis based on a symptom.</span>**

*   **Evidence:** <span style="background:#9254de">The single darkest cell on the map is the intersection of the `method_diagnosis` intent and the `SYMPTOM_AND_DISEASE` entity, with a massive count of **5965**.</span>
*   **Interpretation:** This indicates that the primary use case captured in this dataset is users describing a symptom/disease and asking what it is, what it means, or what they should do next.
    *   *Example Question:* <span style="background:#9254de">"Đau bụng và sốt là dấu hiệu bệnh gì?"</span> (What disease do stomach ache and fever indicate?)

**Observation 3: `drug` and `medical_procedure` entities are most often associated with questions about diagnosis and treatment.**

*   **Evidence:** Looking at the `drug` and `medical_procedure` columns, the highest counts are consistently in the `method_diagnosis` and `treatment` rows.
    *   `drug` appears **499** times with `method_diagnosis` and **234** times with `treatment`.
    *   `medical_procedure` appears **1131** times with `method_diagnosis` and **420** times with `treatment`.
*   **Interpretation:** <span style="background:#9254de">Users are either asking if a specific drug/procedure is the right one (`method_diagnosis`) or asking what drug/procedure to use (`treatment`).</span>
    *   *Example (`method_diagnosis`):* "Có nên dùng thuốc Paracetamol không?" (Should I use Paracetamol?)
    *   *Example (`treatment`):* "Viêm họng uống thuốc gì?" (What medicine to take for a sore throat?)

**Observation 4: Questions about `cause` and `severity` are less likely to mention specific drugs or procedures.**

*   **Evidence:** The cells for `cause` and `severity` intents are very "cold" (light-colored, low numbers) in the `drug` and `medical_procedure` columns. For instance, `cause` co-occurs with `drug` only 47 times.
*   **Interpretation:** This is also logical. When asking about the cause of a symptom, users are less likely to mention a specific treatment. Similarly, when asking how severe a symptom is, they are focused on the symptom itself.

Implications for a Machine Learning Model

*   **Powerful Features for Joint Models:** The strong correlations shown in this heatmap suggest that a **joint intent classification and NER model** would likely perform very well. The model can learn that if it detects a `drug` entity, the sentence's intent is highly unlikely to be `cause`.
*   **Error Analysis Guide:** This heatmap is an excellent tool for debugging. If your model predicts that a sentence with a `drug` has the intent `cause`, you know this is a rare and likely incorrect prediction, pointing you to a specific area for improvement.
*   **Confirms Data Quality:** The logical patterns (e.g., `treatment` intent co-occurring with `drug` and `medical_procedure`) validate that the dataset is well-annotated and reflects real-world user behavior.