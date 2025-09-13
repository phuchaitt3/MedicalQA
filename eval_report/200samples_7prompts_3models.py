import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.linear_model import LinearRegression
import numpy as np

# Create the DataFrame from the provided data
data = {
    'Model & Prompt Strategy': [
        'arcee-ai/Arcee-VyLinh (Extract_VI)', 'arcee-ai/Arcee-VyLinh (Current_Best_VI)', 'vilm/vietcuna-3b-v2 (Extract_VI)',
        'vilm/vietcuna-3b-v2 (Current_Best_VI)', 'vilm/vietcuna-3b-v2 (Expert_Persona_VI)', 'vilm/vietcuna-3b-v2 (RolePlay_VI)',
        'vilm/vietcuna-3b-v2 (Direct_VI)', 'arcee-ai/Arcee-VyLinh (Few_Shot_VI_ViMedAQA)', 'vilm/vietcuna-3b-v2 (Chain_of_Thought_VI)',
        'alpha-ai/LLAMA3-3B-Medical-COT (Current_Best_VI)', 'arcee-ai/Arcee-VyLinh (RolePlay_VI)', 'arcee-ai/Arcee-VyLinh (Direct_VI)',
        'arcee-ai/Arcee-VyLinh (Expert_Persona_VI)', 'vilm/vietcuna-3b-v2 (Few_Shot_VI_ViMedAQA)', 'alpha-ai/LLAMA3-3B-Medical-COT (Direct_VI)',
        'alpha-ai/LLAMA3-3B-Medical-COT (Extract_VI)', 'alpha-ai/LLAMA3-3B-Medical-COT (Few_Shot_VI_Vi...)',
        'alpha-ai/LLAMA3-3B-Medical-COT (RolePlay_VI)', 'alpha-ai/LLAMA3-3B-Medical-COT (Expert_Persona...)',
        'arcee-ai/Arcee-VyLinh (Chain_of_Thought_VI)', 'alpha-ai/LLAMA3-3B-Medical-COT (Chain_of_Thoug...)'
    ],
    'ROUGE-L': [
        0.6214, 0.5832, 0.5403, 0.5333, 0.5284, 0.5195, 0.5167, 0.5262, 0.5061, 0.4925, 0.4841, 0.4692, 0.4263, 0.4399, 0.4300,
        0.4204, 0.4290, 0.4047, 0.3658, 0.2031, 0.1882
    ],
    'BLEU': [
        0.4536, 0.3539, 0.2652, 0.2617, 0.2444, 0.2326, 0.2464, 0.2083, 0.2063, 0.2736, 0.1612, 0.1543, 0.1270, 0.1191, 0.1325,
        0.2236, 0.1427, 0.1078, 0.0932, 0.0683, 0.0578
    ],
    'METEOR': [
        0.5896, 0.5556, 0.5315, 0.5413, 0.5415, 0.5312, 0.5123, 0.6351, 0.5275, 0.4527, 0.6061, 0.6006, 0.5813, 0.4922, 0.4688,
        0.3429, 0.4800, 0.4357, 0.4206, 0.4278, 0.3517
    ],
    'BERTScore-F1': [
        0.8551, 0.8365, 0.8300, 0.8299, 0.8274, 0.8247, 0.8229, 0.8207, 0.8175, 0.8124, 0.8099, 0.8051, 0.7935, 0.7781, 0.7766,
        0.7696, 0.7696, 0.7651, 0.7534, 0.6903, 0.6698
    ],
    'Generation Time (s)': [
        754.84, 897.20, 556.98, 595.47, 578.64, 595.33, 557.12, 1777.16, 674.53, 307.17, 1895.57, 2014.53, 2283.51, 1270.46, 592.89,
        264.47, 674.26, 633.57, 756.64, 4570.40, 1580.11
    ]
}
df = pd.DataFrame(data)

# Extract Model and Prompt Strategy into separate columns
df[['Model', 'Prompt Strategy']] = df['Model & Prompt Strategy'].str.extract(r'([^\(]+)\s\((.*)\)')
model_order = [
    'vilm/vietcuna-3b-v2',
    'arcee-ai/Arcee-VyLinh',
    'alpha-ai/LLAMA3-3B-Medical-COT'
]
df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)

# --- Sắp xếp DataFrame để nhóm theo Model ---
# Sắp xếp theo Model, sau đó theo BERTScore-F1 giảm dần để dễ đọc hơn
df = df.sort_values(by=['Model', 'BERTScore-F1'], ascending=[True, False]).reset_index(drop=True)

# --- Create a new directory for results ---
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
script_name_base = os.path.splitext(os.path.basename(script_path))[0]
results_dir = os.path.join(script_dir, script_name_base)
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved in: {results_dir}")

# --- Save Summary Tables to a Markdown file within the new directory ---
report_filename = os.path.join(results_dir, 'performance_report.md')
with open(report_filename, 'w', encoding='utf-8') as f: # Thêm encoding='utf-8' để hỗ trợ tiếng Việt
    f.write("# Báo cáo hiệu suất các mô hình ngôn ngữ\n\n")

    # --- THÊM PHẦN TÓM TẮT MỚI ---
    f.write("## Tóm tắt hiệu suất theo từng LLM\n\n")
    # Lặp qua từng model để tìm prompt tốt nhất và tệ nhất
    # Dùng df['Model'].unique() đã được sắp xếp để đảm bảo thứ tự
    for model_name in df['Model'].unique():
        model_df = df[df['Model'] == model_name]
        best_prompt_row = model_df.loc[model_df['BERTScore-F1'].idxmax()]
        worst_prompt_row = model_df.loc[model_df['BERTScore-F1'].idxmin()]

        f.write(f"### {model_name.strip()}\n\n")
        f.write(f"*   **Prompt hiệu quả nhất:** `{best_prompt_row['Prompt Strategy']}` với BERTScore-F1 là **{best_prompt_row['BERTScore-F1']:.4f}**.\n")
        f.write(f"*   **Prompt kém hiệu quả nhất:** `{worst_prompt_row['Prompt Strategy']}` với BERTScore-F1 là **{worst_prompt_row['BERTScore-F1']:.4f}**.\n\n")

    # Average performance by model
    avg_performance_by_model = df.groupby('Model', observed=False).mean(numeric_only=True)
    f.write("## Hiệu suất trung bình theo Model\n\n")
    f.write(avg_performance_by_model.to_markdown())
    f.write("\n\n")

    # Best performing prompt strategy for each model based on BERTScore-F1
    best_prompt_by_model = df.loc[df.groupby('Model', observed=False)['BERTScore-F1'].idxmax()]
    f.write("## Prompt hiệu quả nhất theo từng Model (dựa trên BERTScore-F1)\n\n")
    f.write(best_prompt_by_model[['Model', 'Prompt Strategy', 'BERTScore-F1', 'Generation Time (s)']].to_markdown())
    f.write("\n\n")

    # Add links to the saved graphs
    f.write("## Biểu đồ trực quan\n\n")
    f.write(f"### Điểm BERTScore-F1 theo Model và Prompt\n")
    # Sử dụng os.path.basename để lấy tên file cho đường dẫn tương đối trong Markdown
    graph1_basename = os.path.basename(os.path.join(results_dir, 'bertscore_f1_scores_barplot.png'))
    f.write(f"![Biểu đồ cột điểm BERTScore-F1]({graph1_basename})\n\n")

    f.write(f"### Thời gian tạo và điểm BERTScore-F1 (có đường hồi quy)\n")
    graph2_basename = os.path.basename(os.path.join(results_dir, 'generation_time_vs_bertscore_regression.png'))
    f.write(f"![Biểu đồ phân tán thời gian và điểm BERTScore-F1]({graph2_basename})\n\n")


print(f"Báo cáo Markdown đã được lưu tại: {report_filename}")

# --- Add small gaps between the bars of each model ---
# Create a list to hold all the DataFrame pieces
dfs_to_concat = []
for i, model_name in enumerate(model_order):
    # Add the model's DataFrame to the list
    model_df = df[df['Model'] == model_name].copy()
    dfs_to_concat.append(model_df)

    # Add a separator after each model except the last one
    if i < len(model_order) - 1:
        # Create the separator row as a dictionary
        separator_row_data = {col: np.nan for col in df.columns}
        separator_row_data['Model & Prompt Strategy'] = ' ' * (i + 1)
        separator_row_data['Prompt Strategy'] = 'SEPARATOR'
        
        # Convert the dictionary to a single-row DataFrame
        separator_df = pd.DataFrame([separator_row_data])
        
        # KEY CHANGE: Force the separator DataFrame to have the same dtypes as the original.
        # This is the definitive fix for the FutureWarning.
        separator_df = separator_df.astype(df.dtypes)
        
        # Add the correctly-typed separator DataFrame to the list
        dfs_to_concat.append(separator_df)

# Perform a single, efficient concatenation of all pieces. The warning will now be gone.
df_with_gaps = pd.concat(dfs_to_concat, ignore_index=True)

# Re-apply categorical type just to be safe, although it should be preserved
df_with_gaps['Model'] = pd.Categorical(df_with_gaps['Model'], categories=model_order, ordered=True)

# --- Generate and Save Graphs within the new directory ---
# Bar chart of BERTScore-F1 scores by Model and Prompt Strategy
plt.figure(figsize=(12, 10)) # Tăng chiều cao để vừa với các nhãn
ax = sns.barplot(
    data=df_with_gaps, # df đã được sắp xếp
    x='BERTScore-F1',
    y='Model & Prompt Strategy',
    hue='Model', # Tô màu theo Model để dễ phân biệt
    dodge=False, # Tắt dodge để các thanh thẳng hàng
    palette='viridis'
)
plt.title('Điểm BERTScore-F1 theo Model và Prompt')
plt.xlabel('Điểm BERTScore-F1')
plt.ylabel('Model & Prompt Strategy')
plt.legend(title='Model', bbox_to_anchor=(1, -0.05), loc='upper right') # Đã thay đổi
plt.tight_layout()

# Identify the lowest performing prompt for each model
lowest_prompts_per_model_full_string = df.loc[df.groupby('Model', observed=False)['BERTScore-F1'].idxmin()]['Model & Prompt Strategy'].tolist()

# Iterate through the y-axis tick labels and color them
for label_text in ax.get_yticklabels():
    full_label_text = label_text.get_text()

    # Extract Prompt Strategy from the full label text
    start_index = full_label_text.find('(')
    end_index = full_label_text.find(')')
    
    if start_index != -1 and end_index != -1:
        prompt_strategy = full_label_text[start_index + 1 : end_index]
        
        # Apply coloring based on specific prompt strategies
        if prompt_strategy == 'Extract_VI':
            label_text.set_color('darkgreen')
        elif prompt_strategy == 'Current_Best_VI':
            label_text.set_color('blue')
        # Apply coloring for the lowest prompt for each model
        elif full_label_text in lowest_prompts_per_model_full_string:
            label_text.set_color('red')

graph1_filename = os.path.join(results_dir, 'bertscore_f1_scores_barplot.png')
plt.savefig(graph1_filename)
plt.close()
print(f"Biểu đồ '{graph1_filename}' đã được lưu.")


# Scatter plot of Generation Time vs. BERTScore-F1 with Regression Lines
plt.figure(figsize=(10, 6))
ax = sns.scatterplot(data=df, x='Generation Time (s)', y='BERTScore-F1', hue='Model', style='Prompt Strategy', s=100)

for model_name in df['Model'].unique():
    model_df = df[df['Model'] == model_name]
    if len(model_df) > 1:
        sns.regplot(
            data=model_df,
            x='Generation Time (s)',
            y='BERTScore-F1',
            scatter=False,
            ci=None,
            ax=ax,
            line_kws={'alpha': 0.2}
        )

plt.title('Thời gian tạo và điểm BERTScore-F1 (có đường hồi quy)')
plt.xlabel('Thời gian tạo (s)')
plt.ylabel('Điểm BERTScore-F1')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
graph2_filename = os.path.join(results_dir, 'generation_time_vs_bertscore_regression.png')
plt.savefig(graph2_filename)
plt.close()
print(f"Biểu đồ '{graph2_filename}' đã được lưu.")