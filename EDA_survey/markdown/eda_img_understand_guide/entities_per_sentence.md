![[entities_per_sentence.png]]
General Rules for Understanding This Type of Chart

This chart visualizes the **entity density** or **informational complexity** of the sentences in your dataset.

*   **X-axis (Number of Entities):** This represents the count of distinct labeled entities within a single sentence. A value of '0' means the sentence has no labeled entities, '1' means it has one, '2' means two, and so on.
*   **Y-axis (Count of Sentences):** This shows how many sentences in the dataset have the corresponding number of entities from the x-axis.

**When analyzing this chart, you are generally looking for:**

1.  **The Peak (Mode):** <span style="background:#9254de">Which bar is the tallest? This tells you the most common number of entities in a typical sentence.</span>
2.  <span style="background:#9254de">**The "Zero" Bar:**</span> How many sentences have no entities at all? A large number here might indicate many general or irrelevant sentences. A small number suggests the dataset is rich with relevant information. [<1>]
3.  **The Shape of the Distribution:** <span style="background:rgba(240, 200, 0, 0.2)">How quickly do the bars decrease in height as the number of entities increases</span>?
    *   <span style="background:#9254de">A **steep drop-off** means most sentences are simple, focusing on just a few concepts.</span>
    *   <span style="background:#9254de">A **long tail** (bars for higher numbers, like 4, 5, 6+, are still visible) indicates the presence of complex, information-dense sentences.</span>

---

Specific Observations and Interpretation for the ViMQ Dataset

Based on your chart and the statistical report, here are the specific conclusions we can draw about the ViMQ dataset:

**Observation 1: The vast majority of questions are focused and information-rich.**

*   **Evidence:** <span style="background:rgba(205, 244, 105, 0.55)">The bar for **0 entities** is very small</span> (around 250-300 sentences out of 9000). This is an excellent sign. It means that almost every sentence in the dataset contains at least one labeled medical entity, making the dataset highly relevant for NER tasks.

**Observation 2: The most common type of question involves a single medical entity.**

*   **Evidence:** The bar for **1 entity** is the tallest by a large margin, reaching almost 4,700 sentences. <span style="background:#9254de">This tells us that a "typical" question in this dataset is straightforward, like *"Bệnh trĩ có nguy hiểm không?"* (Is hemorrhoid dangerous?), which contains only one `SYMPTOM_AND_DISEASE` entity.</span>

**Observation 3: Questions with two entities are also very common, but complexity quickly decreases after that.**

*   **Evidence:** The bar for **2 entities** is the second-tallest (around 2,750 sentences). Sentences with <span style="background:#9254de">one or two entities make up the vast majority of the dataset. The bars for 3, 4, and 5 entities drop off sharply, and sentences with 6, 7, or 8 entities are extremely rare</span> (the `max` is 8, but the bars are barely visible). This pattern is known as a **right-skewed distribution**.

**Observation 4: The average sentence is relatively simple.**

*   **Evidence:** Your report shows <span style="background:#9254de">the **mean number of entities is 1.64**, and the **median (50th percentile) is 1.0**. This statistically confirms what the chart shows visually: half of all sentences in the dataset have just one entity or fewer</span>. The mean is slightly higher than the median because the few highly complex sentences (with 4+ entities) pull the average up.
[Left off]
Implications for a Machine Learning Model

These observations have direct consequences for building an NER model:

*   **Good for Training:** The model will have plenty of examples to learn what single and double-entity sentences look like.
*   **Challenge with Complexity:** The model will see very few examples of sentences with 4 or more entities. As a result, it might perform poorly on complex, real-world user queries that contain multiple symptoms, drugs, and procedures.
*   **Evaluation Strategy:** When evaluating the model's performance, you shouldn't just look at the overall accuracy. It's crucial to break down the results by the number of entities per sentence. This will reveal if your model is only good at simple cases or if it can handle complex questions too.

# [1.] Zero entities
You're hitting on a crucial nuance in how "relevance" is defined within the context of an NLP dataset, especially one focused on specific tasks like Named Entity Recognition (NER).

When we say a sentence with "zero entities" *might* indicate an irrelevant sentence, we're talking about its relevance to the **core information extraction task of NER** or the direct medical content.

Here's a breakdown of why this is the case, and why it's not always a hard rule:

Why "Zero Entities" *Might* Mean Irrelevant (for NER)

1.  **No Extractable Medical Information for NER:**
    *   The primary goal of NER in a medical dataset like ViMQ is to identify specific categories: `SYMPTOM_AND_DISEASE`, `medical_procedure`, `drug`.
    *   <span style="background:#9254de">If a sentence has zero of these entities, it means there's **nothing for the NER model to extract** in terms of these predefined medical concepts</span>. From the perspective of the NER task, the sentence doesn't directly contribute to identifying core medical facts.

2.  **Examples of "Zero Entity" Sentences (and their nuanced relevance):**

    *   **Purely Conversational/Greeting:**
        *   <span style="background:#9254de">"Chào bác sĩ." (Hello doctor.)</span>
        *   <span style="background:#9254de">"Cảm ơn bạn." (Thank you.)</span>
        *   *Relevance:* Highly relevant for natural dialogue flow in a chatbot, but irrelevant for *extracting medical entities*.
	[Left off]
    *   **General Non-Medical Questions/Statements:**
        *   "Bệnh viện mở cửa mấy giờ?" (What time does the hospital open?)
        *   "Tôi có thể đặt lịch hẹn không?" (Can I book an appointment?)
        *   *Relevance:* Relevant for administrative functions of a healthcare system, but again, no direct medical entity to extract.

    *   **Vague or Ambiguous Statements (depending on annotation guidelines):**
        *   "Tôi cảm thấy không khỏe." (I feel unwell.)
        *   "Tình hình của tôi thế nào rồi?" (How is my condition?)
        *   *Relevance:* These are clearly related to health, but if "không khỏe" isn't explicitly defined as a `SYMPTOM_AND_DISEASE` in the annotation guidelines, it would have 0 entities for NER. For Intent Classification, however, these would likely be classified as a `method_diagnosis` or `severity` intent, indicating their clear relevance.

Why "Irrelevant" Isn't Always the Best Word (in a Broader Context)

The caveat is important: "irrelevant" is a strong word. For a full-fledged dialogue system (like a medical chatbot), many sentences that contain zero NER entities are still **highly relevant** for the overall user experience and dialogue management.

*   A chatbot needs to understand greetings, administrative requests, and user feedback, even if those sentences don't contain a `drug` or `SYMPTOM_AND_DISEASE`.
*   The `sent_label` (intent classification) part of your dataset already accounts for this broader relevance by categorizing such sentences (e.g., a greeting might have an intent label like `greeting`).

Conclusion for ViMQ Specifically

For your ViMQ dataset, the fact that the "0 entities" bar is small (indicating a low count of such sentences) is generally a **positive sign**. It suggests:

*   The annotators largely focused on extracting specific medical information.
*   The dataset is dense with valuable examples for the NER task.
*   There aren't many "empty" or truly irrelevant sentences that would dilute the training data for the core NLP tasks.

In summary, "zero entities" means a sentence doesn't contribute directly to the NER task. While it might be "irrelevant" to *that specific sub-task*, it could still be highly "relevant" to the broader goal of building a functional dialogue system. Your ViMQ dataset appears to be well-curated with a high density of medically relevant entities per sentence.