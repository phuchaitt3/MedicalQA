import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.linear_model import LinearRegression
import numpy as np

# --- CONTROL TOGGLES ---
# Set these to True to run a specific feature, or False to skip it.
# This is useful for rerunning the script to add new features without
# regenerating existing results.
RUN_ALL = True  # If True, overrides all other toggles to True
if RUN_ALL:
    GENERATE_MARKDOWN_REPORT = True
    GENERATE_BAR_CHART = True
    GENERATE_SCATTER_PLOT = True
    GENERATE_PROMPT_ANALYSIS_GRAPH = True
else:
    GENERATE_MARKDOWN_REPORT = True
    GENERATE_BAR_CHART = False
    GENERATE_SCATTER_PLOT = False
    GENERATE_PROMPT_ANALYSIS_GRAPH = True

# --- DATA SETUP ---
# Create the DataFrame from the provided data
# data = {
#     'Model & Prompt Strategy': [
#         'arcee-ai/Arcee-VyLinh (Extract_VI)', 'arcee-ai/Arcee-VyLinh (Current_Best_VI)', 'vilm/vietcuna-3b-v2 (Extract_VI)',
#         'vilm/vietcuna-3b-v2 (Current_Best_VI)', 'vilm/vietcuna-3b-v2 (Expert_Persona_VI)', 'vilm/vietcuna-3b-v2 (RolePlay_VI)',
#         'vilm/vietcuna-3b-v2 (Direct_VI)', 'arcee-ai/Arcee-VyLinh (Few_Shot_VI_ViMedAQA)', 'vilm/vietcuna-3b-v2 (Chain_of_Thought_VI)',
#         'alpha-ai/LLAMA3-3B-Medical-COT (Current_Best_VI)', 'arcee-ai/Arcee-VyLinh (RolePlay_VI)', 'arcee-ai/Arcee-VyLinh (Direct_VI)',
#         'arcee-ai/Arcee-VyLinh (Expert_Persona_VI)', 'vilm/vietcuna-3b-v2 (Few_Shot_VI_ViMedAQA)', 'alpha-ai/LLAMA3-3B-Medical-COT (Direct_VI)',
#         'alpha-ai/LLAMA3-3B-Medical-COT (Extract_VI)', 'alpha-ai/LLAMA3-3B-Medical-COT (Few_Shot_VI_Vi...)',
#         'alpha-ai/LLAMA3-3B-Medical-COT (RolePlay_VI)', 'alpha-ai/LLAMA3-3B-Medical-COT (Expert_Persona...)',
#         'arcee-ai/Arcee-VyLinh (Chain_of_Thought_VI)', 'alpha-ai/LLAMA3-3B-Medical-COT (Chain_of_Thoug...)'
#     ],
#     'ROUGE-L': [
#         0.6214, 0.5832, 0.5403, 0.5333, 0.5284, 0.5195, 0.5167, 0.5262, 0.5061, 0.4925, 0.4841, 0.4692, 0.4263, 0.4399, 0.4300,
#         0.4204, 0.4290, 0.4047, 0.3658, 0.2031, 0.1882
#     ],
#     'BLEU': [
#         0.4536, 0.3539, 0.2652, 0.2617, 0.2444, 0.2326, 0.2464, 0.2083, 0.2063, 0.2736, 0.1612, 0.1543, 0.1270, 0.1191, 0.1325,
#         0.2236, 0.1427, 0.1078, 0.0932, 0.0683, 0.0578
#     ],
#     'METEOR': [
#         0.5896, 0.5556, 0.5315, 0.5413, 0.5415, 0.5312, 0.5123, 0.6351, 0.5275, 0.4527, 0.6061, 0.6006, 0.5813, 0.4922, 0.4688,
#         0.3429, 0.4800, 0.4357, 0.4206, 0.4278, 0.3517
#     ],
#     'BERTScore-F1': [
#         0.8551, 0.8365, 0.8300, 0.8299, 0.8274, 0.8247, 0.8229, 0.8207, 0.8175, 0.8124, 0.8099, 0.8051, 0.7935, 0.7781, 0.7766,
#         0.7696, 0.7696, 0.7651, 0.7534, 0.6903, 0.6698
#     ],
#     'Generation Time (s)': [
#         754.84, 897.20, 556.98, 595.47, 578.64, 595.33, 557.12, 1777.16, 674.53, 307.17, 1895.57, 2014.53, 2283.51, 1270.46, 592.89,
#         264.47, 674.26, 633.57, 756.64, 4570.40, 1580.11
#     ]
# }

# new_data_rows = [
#     {'Model & Prompt Strategy': 'sail/Sailor-4B-Chat (Chain_of_Thought_VI)', 'ROUGE-L': 0.1858, 'BLEU': 0.0618, 'METEOR': 0.3941, 'BERTScore-F1': 0.6781, 'Generation Time (s)': 4986.24},
#     {'Model & Prompt Strategy': 'sail/Sailor-4B-Chat (Direct_VI)', 'ROUGE-L': 0.2014, 'BLEU': 0.0697, 'METEOR': 0.4242, 'BERTScore-F1': 0.6627, 'Generation Time (s)': 4993.69},
#     {'Model & Prompt Strategy': 'sail/Sailor-4B-Chat (Current_Best_VI)', 'ROUGE-L': 0.2100, 'BLEU': 0.0730, 'METEOR': 0.4281, 'BERTScore-F1': 0.6611, 'Generation Time (s)': 5003.86},
#     {'Model & Prompt Strategy': 'sail/Sailor-4B-Chat (Expert_Persona_VI)', 'ROUGE-L': 0.1991, 'BLEU': 0.0664, 'METEOR': 0.4192, 'BERTScore-F1': 0.6611, 'Generation Time (s)': 4982.71},
#     {'Model & Prompt Strategy': 'sail/Sailor-4B-Chat (RolePlay_VI)', 'ROUGE-L': 0.2022, 'BLEU': 0.0686, 'METEOR': 0.4210, 'BERTScore-F1': 0.6593, 'Generation Time (s)': 4979.34},
#     {'Model & Prompt Strategy': 'sail/Sailor-4B-Chat (Extract_VI)', 'ROUGE-L': 0.2125, 'BLEU': 0.0760, 'METEOR': 0.4425, 'BERTScore-F1': 0.6555, 'Generation Time (s)': 4977.39},
#     {'Model & Prompt Strategy': 'sail/Sailor-4B-Chat (Few_Shot_VI_ViMedAQA)', 'ROUGE-L': 0.2086, 'BLEU': 0.0715, 'METEOR': 0.4124, 'BERTScore-F1': 0.6457, 'Generation Time (s)': 5227.87}
# ]

# for row in new_data_rows:
#     data['Model & Prompt Strategy'].append(row['Model & Prompt Strategy'])
#     data['ROUGE-L'].append(row['ROUGE-L'])
#     data['BLEU'].append(row['BLEU'])
#     data['METEOR'].append(row['METEOR'])
#     data['BERTScore-F1'].append(row['BERTScore-F1'])
#     data['Generation Time (s)'].append(row['Generation Time (s)'])

data = {
    'Model & Prompt Strategy': [
        'vilm/vinallama-2.7b-chat (Extract_VI)',
        'vilm/vinallama-2.7b-chat (Current_Best_VI)',
        'vilm/vinallama-2.7b-chat (Direct_VI)',
        'vilm/vinallama-2.7b-chat (Few_Shot_VI_ViMedAQA)',
        'vilm/vietcuna-3b-v2 (Extract_VI)',
        'vilm/vietcuna-3b-v2 (Current_Best_VI)',
        'vilm/vietcuna-3b-v2 (Direct_VI)',
        'vilm/vietcuna-3b-v2 (Chain_of_Thought_VI)',
        'vilm/vinallama-2.7b-chat (RolePlay_VI)',
        'vilm/vietcuna-3b-v2 (Expert_Persona_VI)',
        'vilm/vinallama-2.7b-chat (Expert_Persona_VI)',
        'vilm/vietcuna-3b-v2 (RolePlay_VI)',
        'vilm/vietcuna-3b-v2 (Few_Shot_VI_ViMedAQA)',
        'sail/Sailor-4B (Current_Best_VI)',
        'sail/Sailor-4B (Direct_VI)',
        'sail/Sailor-4B (Chain_of_Thought_VI)',
        'sail/Sailor-4B (RolePlay_VI)',
        'vilm/vinallama-2.7b-chat (Chain_of_Thought_VI)',
        'sail/Sailor-4B (Few_Shot_VI_ViMedAQA)',
        'sail/Sailor-4B (Expert_Persona_VI)',
        'sail/Sailor-4B (Extract_VI)'
    ],
    'ROUGE-L': [
        0.6013, 0.5659, 0.5139, 0.5131, 0.4933, 0.4564, 0.4623, 0.4535, 0.4415, 0.4395, 0.4240, 0.4407, 0.4465, 0.3422,
        0.3118, 0.3203, 0.3106, 0.2299, 0.2345, 0.2861, 0.2510
    ],
    'BLEU': [
        0.3641, 0.2459, 0.2523, 0.2686, 0.1700, 0.1643, 0.1388, 0.1489, 0.1570, 0.1377, 0.1822, 0.1091, 0.1204, 0.1010,
        0.0896, 0.0961, 0.0880, 0.0805, 0.0760, 0.0893, 0.0725
    ],
    'METEOR': [
        0.6483, 0.5926, 0.6524, 0.7057, 0.5234, 0.4927, 0.4899, 0.5033, 0.5867, 0.5008, 0.5720, 0.4840, 0.5402, 0.5399,
        0.5030, 0.5242, 0.4915, 0.4719, 0.4278, 0.4863, 0.4257
    ],
    'BERTScore-F1': [
        0.8570, 0.8346, 0.8320, 0.8259, 0.8138, 0.8117, 0.8094, 0.8045, 0.7966, 0.7963, 0.7952, 0.7881, 0.7798, 0.7042,
        0.6971, 0.6952, 0.6905, 0.6884, 0.6789, 0.6704, 0.6646
    ],
    'Generation Time (s)': [
        163.50, 164.05, 163.19, 171.41, 34.33, 37.26, 36.71, 38.42, 163.06, 41.62, 162.91, 48.00, 61.83, 169.63,
        182.62, 182.32, 184.13, 163.29, 211.54, 190.41, 200.11
    ]
}
    
df = pd.DataFrame(data)

# Extract Model and Prompt Strategy into separate columns
df[['Model', 'Prompt Strategy']] = df['Model & Prompt Strategy'].str.extract(r'([^\(]+)\s\((.*)\)')
model_order = [
    'vilm/vinallama-2.7b-chat',
    'vilm/vietcuna-3b-v2',
    # 'arcee-ai/Arcee-VyLinh',
    # 'alpha-ai/LLAMA3-3B-Medical-COT',
    'sail/Sailor-4B',
]
df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)

# Sort DataFrame to group by Model, then by BERTScore-F1 descending
df = df.sort_values(by=['Model', 'BERTScore-F1'], ascending=[True, False]).reset_index(drop=True)

# --- DIRECTORY SETUP ---
# Create a new directory for results
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
script_name_base = os.path.splitext(os.path.basename(script_path))[0]
results_dir = os.path.join(script_dir, script_name_base)
os.makedirs(results_dir, exist_ok=True)

# --- FEATURE 1: GENERATE MARKDOWN REPORT ---
if GENERATE_MARKDOWN_REPORT:
    report_filename = os.path.join(results_dir, 'performance_report.md')
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("# Báo cáo hiệu suất các mô hình ngôn ngữ\n\n")

        # ... (previous report sections) ...

        best_prompt_by_model = df.loc[df.groupby('Model', observed=False)['BERTScore-F1'].idxmax()]
        f.write("## Prompt hiệu quả nhất theo từng Model (dựa trên BERTScore-F1)\n\n")
        f.write(best_prompt_by_model[['Model', 'Prompt Strategy', 'BERTScore-F1', 'Generation Time (s)']].to_markdown())
        f.write("\n\n")

        # --- MODIFIED SECTION ---
        # Add the prompt strategy table to the report
        avg_by_prompt = df.groupby('Prompt Strategy').mean(numeric_only=True)
        prompt_counts = df.groupby('Prompt Strategy').size().rename('Usage Count')
        prompt_analysis = pd.concat([avg_by_prompt, prompt_counts], axis=1)
        prompt_analysis = prompt_analysis.sort_values(by='BERTScore-F1', ascending=False)
        f.write("## Hiệu suất trung bình theo Prompt Strategy (Bảng)\n\n")
        f.write(prompt_analysis.to_markdown())
        f.write("\n\n")
        # --- END MODIFIED SECTION ---

        f.write("## Biểu đồ trực quan\n\n")

        # Conditionally add links to graphs
        if GENERATE_BAR_CHART:
            f.write(f"### Điểm BERTScore-F1 theo Model và Prompt\n")
            graph1_basename = os.path.basename('bertscore_f1_scores_barplot.png')
            f.write(f"![Biểu đồ cột điểm BERTScore-F1]({graph1_basename})\n\n")

        if GENERATE_SCATTER_PLOT:
            f.write(f"### Thời gian tạo và điểm BERTScore-F1 (có đường hồi quy)\n")
            graph2_basename = os.path.basename('generation_time_vs_bertscore_regression.png')
            f.write(f"![Biểu đồ phân tán thời gian và điểm BERTScore-F1]({graph2_basename})\n\n")
        
        # --- ADDED SECTION ---
        if GENERATE_PROMPT_ANALYSIS_GRAPH:
            f.write(f"### Hiệu suất trung bình theo Prompt Strategy (Biểu đồ)\n")
            graph3_basename = os.path.basename('prompt_strategy_performance.png')
            f.write(f"![Biểu đồ hiệu suất Prompt Strategy]({graph3_basename})\n\n")
        # --- END ADDED SECTION ---

    print(f"Báo cáo Markdown đã được lưu tại: {report_filename}")

# --- FEATURE 2: GENERATE BAR CHART ---
if GENERATE_BAR_CHART:
    # Add small gaps between the bars of each model for visual separation
    dfs_to_concat = []
    for i, model_name in enumerate(model_order):
        model_df = df[df['Model'] == model_name].copy()
        dfs_to_concat.append(model_df)

        if i < len(model_order) - 1:
            separator_row_data = {col: np.nan for col in df.columns}
            separator_row_data['Model & Prompt Strategy'] = ' ' * (i + 1)
            separator_row_data['Prompt Strategy'] = 'SEPARATOR'
            separator_df = pd.DataFrame([separator_row_data])
            separator_df = separator_df.astype(df.dtypes)
            dfs_to_concat.append(separator_df)

    df_with_gaps = pd.concat(dfs_to_concat, ignore_index=True)
    df_with_gaps['Model'] = pd.Categorical(df_with_gaps['Model'], categories=model_order, ordered=True)

    # Bar chart of BERTScore-F1 scores
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(
        data=df_with_gaps,
        x='BERTScore-F1',
        y='Model & Prompt Strategy',
        hue='Model',
        dodge=False,
        palette='viridis'
    )
    plt.title('Điểm BERTScore-F1 theo Model và Prompt')
    plt.xlabel('Điểm BERTScore-F1')
    plt.ylabel('Model & Prompt Strategy')
    plt.legend(title='Model', bbox_to_anchor=(1, -0.05), loc='upper right')
    plt.tight_layout()

    lowest_prompts_per_model_full_string = df.loc[df.groupby('Model', observed=False)['BERTScore-F1'].idxmin()]['Model & Prompt Strategy'].tolist()

    for label_text in ax.get_yticklabels():
        full_label_text = label_text.get_text()
        start_index = full_label_text.find('(')
        end_index = full_label_text.find(')')
        if start_index != -1 and end_index != -1:
            prompt_strategy = full_label_text[start_index + 1 : end_index]
            if prompt_strategy == 'Extract_VI':
                label_text.set_color('darkgreen')
            elif prompt_strategy == 'Current_Best_VI':
                label_text.set_color('blue')
            elif full_label_text in lowest_prompts_per_model_full_string:
                label_text.set_color('red')

    graph1_filename = os.path.join(results_dir, 'bertscore_f1_scores_barplot.png')
    plt.savefig(graph1_filename)
    plt.close()
    print(f"Biểu đồ '{graph1_filename}' đã được lưu.")

# --- FEATURE 3: GENERATE SCATTER PLOT ---
if GENERATE_SCATTER_PLOT:
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
    
    # --- FEATURE 4: ANALYZE BY PROMPT STRATEGY ---
if GENERATE_PROMPT_ANALYSIS_GRAPH:
    # Calculate average performance metrics for each prompt strategy
    avg_by_prompt = df.groupby('Prompt Strategy').mean(numeric_only=True)
    # Sort by BERTScore-F1 to have the best-performing prompts at the top
    prompt_analysis_sorted = avg_by_prompt.sort_values(by='BERTScore-F1', ascending=False)

    # Create the bar chart
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        data=prompt_analysis_sorted,
        x='BERTScore-F1',
        y=prompt_analysis_sorted.index, # Use the index (Prompt Strategy names) for the y-axis
        hue='BERTScore-F1',
        dodge=False,
        palette='plasma'
    )
    plt.title('Hiệu suất trung bình theo Prompt Strategy')
    plt.xlabel('Điểm BERTScore-F1 trung bình')
    plt.ylabel('Prompt Strategy')
    plt.tight_layout()

    # Save the graph to a file
    graph3_filename = os.path.join(results_dir, 'prompt_strategy_performance.png')
    plt.savefig(graph3_filename)
    plt.close()
    print(f"Biểu đồ phân tích Prompt Strategy đã được lưu tại: '{graph3_filename}'")