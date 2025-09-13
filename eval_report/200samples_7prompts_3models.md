- ![[bertscore_f1_scores_barplot.png]]

- The arcee-ai/Arcee-VyLinh model, particularly with the Extract_VI prompt strategy, demonstrated superior performance across all language understanding metrics.
	- This model also exhibited some of the longest generation times, particularly with more complex prompting strategies like Chain_of_Thought_VI.
- The alpha-ai/LLAMA3-3B-Medical-COT model consistently offered the fastest generation times.
	- The Extract_VI strategy clocking the fastest time at 264.47 seconds
- The vilm/vietcuna-3b-v2 model provided a more balanced performance, delivering competitive results across most metrics while maintaining a moderate generation time. Its performance was most effective with the Extract_VI and Current_Best_VI prompt strategies.
- A notable trend across all models was the effectiveness of the Extract_VI and Current_Best_VI prompt strategies in achieving higher performance scores. Conversely, the Chain_of_Thought_VI strategy consistently resulted in the lowest performance for all models and, in the case of arcee-ai/Arcee-VyLinh, the longest generation time.
- ![[generation_time_vs_bertscore_regression.png]]
	- For each specific model, the prompt strategies that took longer to generate a response also tended to produce lower-quality results (as measured by BERTScore-F1).
	- Prompting strategies like `Extract_VI` and `Current_Best_VI` are likely more direct and computationally less intensive. They instruct the model to perform a task without much overhead, resulting in faster generation times and higher-quality, more concise answers that align well with the reference text for BERTScore.

# Average Performance by Model

| Model                          |   ROUGE-L |     BLEU |   METEOR |   BERTScore-F1 |   Generation Time (s) |
|:-------------------------------|----------:|---------:|---------:|---------------:|----------------------:|
| alpha-ai/LLAMA3-3B-Medical-COT |  0.390086 | 0.147314 | 0.421771 |       0.7595   |               687.016 |
| arcee-ai/Arcee-VyLinh          |  0.473357 | 0.218086 | 0.570871 |       0.801586 |              2027.6   |
| vilm/vietcuna-3b-v2            |  0.512029 | 0.2251   | 0.525357 |       0.818643 |               689.79  |
1. **`arcee-ai/Arcee-VyLinh`**: The High-Quality, High-Cost Champion
*   **Average Performance:** Its average quality scores are strong, but it is heavily penalized by its **extremely slow average generation time (2027.60 s)**, which is nearly 3 times slower than the other two models.
*   **This model's performance is highly sensitive to the prompt strategy**. While it can produce the best results, it is inefficient on average, likely due to very poor performance with complex prompts (like `Chain_of_Thought_VI`). It is the best choice for tasks where achieving maximum quality is critical and processing time is a secondary concern.

1. **`vilm/vietcuna-3b-v2`**: The Balanced and Efficient All-Rounder
*   **Average Performance:** This model stands out as the most impressive overall. It has the **highest average scores across all quality metrics** (ROUGE-L, BLEU, METEOR, BERTScore-F1) while maintaining an **excellent average generation time (689.79 s)**.
*   **Conclusion:** This model is the most reliable and well-rounded. It consistently delivers high-quality results without a significant speed penalty, making it the best choice for general-purpose applications where both performance and efficiency are important.

1. **`alpha-ai/LLAMA3-3B-Medical-COT`**: The Speed Specialist
*   **Average Performance:** It consistently has the **lowest average quality scores**. Its primary strength is its speed, being tied with `vilm/vietcuna-3b-v2` for the fastest average generation time.
* Generally not recommended for Vietnamese prompts, need further testing with English prompts but for Vietnamese dataset.

# Best Performing Prompt Strategy by Model (based on BERTScore-F1)

|     | Model                          | Prompt Strategy | BERTScore-F1 | Generation Time (s) |
| --: | :----------------------------- | :-------------- | -----------: | ------------------: |
|   9 | alpha-ai/LLAMA3-3B-Medical-COT | Current_Best_VI |       0.8124 |              307.17 |
|   0 | arcee-ai/Arcee-VyLinh          | Extract_VI      |       0.8551 |              754.84 |
|   2 | vilm/vietcuna-3b-v2            | Extract_VI      |         0.83 |              556.98 |
