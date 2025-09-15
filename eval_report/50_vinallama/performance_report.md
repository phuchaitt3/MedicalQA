# Báo cáo hiệu suất các mô hình ngôn ngữ

## Prompt hiệu quả nhất theo từng Model (dựa trên BERTScore-F1)

|    | Model                 | Prompt Strategy   |   BERTScore-F1 |   Generation Time (s) |
|---:|:----------------------|:------------------|---------------:|----------------------:|
|  0 | Arcee-VyLinh          | Extract           |         0.8551 |                754.84 |
|  7 | vietcuna-3b-v2        | Extract           |         0.83   |                556.98 |
| 14 | *vinallama-2.7b-chat  | Extract           |         0.8217 |               3868.56 |
| 21 | LLAMA3-3B-Medical-COT | Current_Best      |         0.8124 |                307.17 |
| 28 | Sailor-4B-Chat        | Chain_of_Thought  |         0.6781 |               4986.24 |

## Hiệu suất trung bình theo Prompt Strategy (Bảng)

| Prompt Strategy   |   ROUGE-L |    BLEU |   METEOR |   BERTScore-F1 |   Generation Time (s) |   Usage Count |
|:------------------|----------:|--------:|---------:|---------------:|----------------------:|--------------:|
| Current_Best      |   0.46582 | 0.2396  |  0.50658 |        0.79212 |               2137.69 |             5 |
| Extract           |   0.46242 | 0.26186 |  0.49676 |        0.78638 |               2084.45 |             5 |
| Direct            |   0.41668 | 0.15498 |  0.51542 |        0.77386 |               2423.25 |             5 |
| RolePlay          |   0.40966 | 0.1442  |  0.5097  |        0.77036 |               2399.86 |             5 |
| Few_Shot          |   0.42518 | 0.1558  |  0.52566 |        0.76466 |               2597.3  |             5 |
| Expert_Persona    |   0.38552 | 0.13366 |  0.49658 |        0.76316 |               2489.71 |             5 |
| Chain_of_Thought  |   0.26042 | 0.09334 |  0.42682 |        0.70786 |               3132.75 |             5 |

## Biểu đồ trực quan

### Điểm BERTScore-F1 theo Model và Prompt
![Biểu đồ cột điểm BERTScore-F1](bertscore_f1_scores_barplot.png)

### Generation time và điểm BERTScore-F1 (có đường hồi quy)
![Biểu đồ phân tán thời gian và điểm BERTScore-F1](generation_time_vs_bertscore_regression.png)

### Hiệu suất trung bình theo Prompt Strategy (Biểu đồ)
![Biểu đồ hiệu suất Prompt Strategy](prompt_strategy_performance.png)

