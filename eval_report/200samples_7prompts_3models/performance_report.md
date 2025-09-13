# Báo cáo hiệu suất các mô hình ngôn ngữ

## Tóm tắt hiệu suất theo từng LLM

### vilm/vietcuna-3b-v2

*   **Prompt hiệu quả nhất:** `Extract_VI` với BERTScore-F1 là **0.8300**.
*   **Prompt kém hiệu quả nhất:** `Few_Shot_VI_ViMedAQA` với BERTScore-F1 là **0.7781**.

### arcee-ai/Arcee-VyLinh

*   **Prompt hiệu quả nhất:** `Extract_VI` với BERTScore-F1 là **0.8551**.
*   **Prompt kém hiệu quả nhất:** `Chain_of_Thought_VI` với BERTScore-F1 là **0.6903**.

### alpha-ai/LLAMA3-3B-Medical-COT

*   **Prompt hiệu quả nhất:** `Current_Best_VI` với BERTScore-F1 là **0.8124**.
*   **Prompt kém hiệu quả nhất:** `Chain_of_Thoug...` với BERTScore-F1 là **0.6698**.

## Hiệu suất trung bình theo Model

| Model                          |   ROUGE-L |     BLEU |   METEOR |   BERTScore-F1 |   Generation Time (s) |
|:-------------------------------|----------:|---------:|---------:|---------------:|----------------------:|
| vilm/vietcuna-3b-v2            |  0.512029 | 0.2251   | 0.525357 |       0.818643 |               689.79  |
| arcee-ai/Arcee-VyLinh          |  0.473357 | 0.218086 | 0.570871 |       0.801586 |              2027.6   |
| alpha-ai/LLAMA3-3B-Medical-COT |  0.390086 | 0.147314 | 0.421771 |       0.7595   |               687.016 |

## Prompt hiệu quả nhất theo từng Model (dựa trên BERTScore-F1)

|    | Model                          | Prompt Strategy   |   BERTScore-F1 |   Generation Time (s) |
|---:|:-------------------------------|:------------------|---------------:|----------------------:|
|  0 | vilm/vietcuna-3b-v2            | Extract_VI        |         0.83   |                556.98 |
|  7 | arcee-ai/Arcee-VyLinh          | Extract_VI        |         0.8551 |                754.84 |
| 14 | alpha-ai/LLAMA3-3B-Medical-COT | Current_Best_VI   |         0.8124 |                307.17 |

## Biểu đồ trực quan

### Điểm BERTScore-F1 theo Model và Prompt
![Biểu đồ cột điểm BERTScore-F1](bertscore_f1_scores_barplot.png)

### Thời gian tạo và điểm BERTScore-F1 (có đường hồi quy)
![Biểu đồ phân tán thời gian và điểm BERTScore-F1](generation_time_vs_bertscore_regression.png)

