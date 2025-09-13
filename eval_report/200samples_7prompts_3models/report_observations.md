[[BERTScore-F1 vs ROUGE-L]]
[[performance_report]]
Based on the provided summary tables, here are the key observations about the performance of the three models:

### High-Level Summary

There is a distinct trade-off between performance quality and generation speed among the models. No single model is the best in all categories; instead, each occupies a specific niche: one is the quality champion, one is the balanced all-rounder, and one is the speed specialist.

---

### Detailed Model Analysis

1. **`arcee-ai/Arcee-VyLinh`**: The High-Quality, High-Cost Champion
*   **Average Performance:** Its average quality scores are strong, but it is heavily penalized by its **extremely slow average generation time (2027.60 s)**, which is nearly 3 times slower than the other two models.
*   **This model's performance is highly sensitive to the prompt strategy**. While it can produce the best results, it is inefficient on average, likely due to very poor performance with complex prompts (like `Chain_of_Thought_VI`). It is the best choice for tasks where achieving maximum quality is critical and processing time is a secondary concern.

1. **`vilm/vietcuna-3b-v2`**: The Balanced and Efficient All-Rounder
*   **Average Performance:** This model stands out as the most impressive overall. It has the **highest average scores across all quality metrics** (ROUGE-L, BLEU, METEOR, BERTScore-F1) while maintaining an **excellent average generation time (689.79 s)**.
*   **Conclusion:** This model is the most reliable and well-rounded. It consistently delivers high-quality results without a significant speed penalty, making it the best choice for general-purpose applications where both performance and efficiency are important.

1. **`alpha-ai/LLAMA3-3B-Medical-COT`**: The Speed Specialist
*   **Average Performance:** It consistently has the **lowest average quality scores**. Its primary strength is its speed, being tied with `vilm/vietcuna-3b-v2` for the fastest average generation time.
* Generally not recommended for Vietnamese prompts, need further testing with English prompts but for Vietnamese dataset.

---

### General Observations

*   **Impact of Prompt Strategy:** The "Best Performing" table reveals that simpler, more direct prompt strategies like **`Extract_VI`** and **`Current_Best_VI`** are overwhelmingly more effective. This suggests that for this specific task, complex reasoning or persona-based prompts add overhead that hurts both speed and quality.
*   **Clear Performance Tiers:** The models can be clearly tiered:
    1.  **Top Quality:** `arcee-ai/Arcee-VyLinh` (but slow)
    2.  **Best Balance:** `vilm/vietcuna-3b-v2`
    3.  **Top Speed:** `alpha-ai/LLAMA3-3B-Medical-COT` (but lower quality)