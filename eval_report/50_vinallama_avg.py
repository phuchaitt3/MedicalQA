import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.linear_model import LinearRegression
import numpy as np
import textwrap

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
    GENERATE_MARKDOWN_REPORT = False
    GENERATE_BAR_CHART = False
    GENERATE_SCATTER_PLOT = False
    GENERATE_PROMPT_ANALYSIS_GRAPH = True

# --- DATA SETUP ---
# Create the DataFrame from the provided data
data = {
    'Model & Prompt Strategy': [
        # --- Initial 21 rows ---
        'Arcee-VyLinh (Extract)', 'Arcee-VyLinh (Current_Best)', 'vietcuna-3b-v2 (Extract)',
        'vietcuna-3b-v2 (Current_Best)', 'vietcuna-3b-v2 (Expert_Persona)', 'vietcuna-3b-v2 (RolePlay)',
        'vietcuna-3b-v2 (Direct)', 'Arcee-VyLinh (Few_Shot)', 'vietcuna-3b-v2 (Chain_of_Thought)',
        'LLAMA3-3B-Medical-COT (Current_Best)', 'Arcee-VyLinh (RolePlay)', 'Arcee-VyLinh (Direct)',
        'Arcee-VyLinh (Expert_Persona)', 'vietcuna-3b-v2 (Few_Shot)', 'LLAMA3-3B-Medical-COT (Direct)',
        'LLAMA3-3B-Medical-COT (Extract)', 'LLAMA3-3B-Medical-COT (Few_Shot)',
        'LLAMA3-3B-Medical-COT (RolePlay)', 'LLAMA3-3B-Medical-COT (Expert_Persona)',
        'Arcee-VyLinh (Chain_of_Thought)', 'LLAMA3-3B-Medical-COT (Chain_of_Thought)',
        # --- fifty_samples_data (14 rows) ---
        '*vinallama-2.7b-chat (Extract)', '*vinallama-2.7b-chat (Current_Best)', '*vietcuna-3b-v2 (Current_Best)',
        '*vietcuna-3b-v2 (Extract)', '*vietcuna-3b-v2 (Expert_Persona)', '*vietcuna-3b-v2 (Direct)',
        '*vinallama-2.7b-chat (Few_Shot)', '*vietcuna-3b-v2 (RolePlay)', '*vietcuna-3b-v2 (Chain_of_Thought)',
        '*vinallama-2.7b-chat (Direct)', '*vinallama-2.7b-chat (RolePlay)', '*vinallama-2.7b-chat (Expert_Persona)',
        '*vietcuna-3b-v2 (Few_Shot)', '*vinallama-2.7b-chat (Chain_of_Thought)',
        # --- sailor_4b_chat_rows (7 rows) ---
        'Sailor-4B-Chat (Chain_of_Thought)', 'Sailor-4B-Chat (Direct)', 'Sailor-4B-Chat (Current_Best)',
        'Sailor-4B-Chat (Expert_Persona)', 'Sailor-4B-Chat (RolePlay)', 'Sailor-4B-Chat (Extract)',
        'Sailor-4B-Chat (Few_Shot)',
    ],
    'ROUGE-L': [
        # --- Initial 21 rows ---
        0.6214, 0.5832, 0.5403, 0.5333, 0.5284, 0.5195, 0.5167, 0.5262, 0.5061, 0.4925, 0.4841, 0.4692, 0.4263, 0.4399, 0.4300,
        0.4204, 0.4290, 0.4047, 0.3658, 0.2031, 0.1882,
        # --- fifty_samples_data (14 rows) ---
        0.5175, 0.5101, 0.5069, 0.4982, 0.5008, 0.4826, 0.5222, 0.4699, 0.4676, 0.4661, 0.4378, 0.4080, 0.4136, 0.2189,
        # --- sailor_4b_chat_rows (7 rows) ---
        0.1858, 0.2014, 0.2100, 0.1991, 0.2022, 0.2125, 0.2086,
    ],
    'BLEU': [
        # --- Initial 21 rows ---
        0.4536, 0.3539, 0.2652, 0.2617, 0.2444, 0.2326, 0.2464, 0.2083, 0.2063, 0.2736, 0.1612, 0.1543, 0.1270, 0.1191, 0.1325,
        0.2236, 0.1427, 0.1078, 0.0932, 0.0683, 0.0578,
        # --- fifty_samples_data (14 rows) ---
        0.2909, 0.2358, 0.2337, 0.2461, 0.2429, 0.2229, 0.2374, 0.2058, 0.2246, 0.1720, 0.1508, 0.1373, 0.1309, 0.0725,
        # --- sailor_4b_chat_rows (7 rows) ---
        0.0618, 0.0697, 0.0730, 0.0664, 0.0686, 0.0760, 0.0715,
    ],
    'METEOR': [
        # --- Initial 21 rows ---
        0.5896, 0.5556, 0.5315, 0.5413, 0.5415, 0.5312, 0.5123, 0.6351, 0.5275, 0.4527, 0.6061, 0.6006, 0.5813, 0.4922, 0.4688,
        0.3429, 0.4800, 0.4357, 0.4206, 0.4278, 0.3517,
        # --- fifty_samples_data (14 rows) ---
        0.5773, 0.5552, 0.5210, 0.4952, 0.5047, 0.4922, 0.6086, 0.4791, 0.4876, 0.5712, 0.5545, 0.5203, 0.4662, 0.4330,
        # --- sailor_4b_chat_rows (7 rows) ---
        0.3941, 0.4242, 0.4281, 0.4192, 0.4210, 0.4425, 0.4124,
    ],
    'BERTScore-F1': [
        # --- Initial 21 rows ---
        0.8551, 0.8365, 0.8300, 0.8299, 0.8274, 0.8247, 0.8229, 0.8207, 0.8175, 0.8124, 0.8099, 0.8051, 0.7935, 0.7781, 0.7766,
        0.7696, 0.7696, 0.7651, 0.7534, 0.6903, 0.6698,
        # --- fifty_samples_data (14 rows) ---
        0.8217, 0.8207, 0.8137, 0.8131, 0.8115, 0.8111, 0.8092, 0.8036, 0.8034, 0.8020, 0.7928, 0.7804, 0.7684, 0.6836,
        # --- sailor_4b_chat_rows (7 rows) ---
        0.6781, 0.6627, 0.6611, 0.6611, 0.6593, 0.6555, 0.6457,
    ],
    'Generation Time (s)': [
        # --- Initial 21 rows ---
        754.84, 897.20, 556.98, 595.47, 578.64, 595.33, 557.12, 1777.16, 674.53, 307.17, 1895.57, 2014.53, 2283.51, 1270.46, 592.89,
        264.47, 674.26, 633.57, 756.64, 4570.40, 1580.11,
        # --- fifty_samples_data (14 rows) ---
        967.14*4, 971.19*4, 160.14*4, 141.25*4, 142.76*4, 144.44*4, 1009.19*4, 148.24*4, 148.51*4, 989.50*4, 973.87*4, 961.76*4, 291.61*4, 963.12*4,
        # --- sailor_4b_chat_rows (7 rows) ---
        4986.24, 4993.69, 5003.86, 4982.71, 4979.34, 4977.39, 5227.87,
    ]
}

# You can now create the final DataFrame from this single, complete dictionary
df = pd.DataFrame(data)

# Define the metrics to be averaged
score_columns = ['ROUGE-L', 'BLEU', 'METEOR', 'BERTScore-F1']
# Calculate the mean across these columns and create the 'Avg_Score' column
df['Avg_Score'] = df[score_columns].mean(axis=1)

# Extract Model and Prompt Strategy into separate columns
df[['Model', 'Prompt Strategy']] = df['Model & Prompt Strategy'].str.extract(r'([^\(]+)\s\((.*)\)')
model_order = [
    'Arcee-VyLinh',
    'vietcuna-3b-v2',
    # '*vietcuna-3b-v2',
    '*vinallama-2.7b-chat',
    'LLAMA3-3B-Medical-COT',
    'Sailor-4B-Chat',
]

# First, filter the DataFrame to only include models present in the model_order list.
# This ensures that commented-out models are completely removed from all subsequent analysis.
df = df[df['Model'].isin(model_order)].copy()

# Now, set the filtered 'Model' column as a categorical type with the specified order.
# This correctly orders the remaining models for plotting and grouping.
df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)

# Sort DataFrame to group by Model, then by Avg_Score descending
df = df.sort_values(by=['Model', 'Avg_Score'], ascending=[True, False]).reset_index(drop=True)

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

        # MODIFIED: Use 'Avg_Score' to find the best prompt
        best_prompt_by_model = df.loc[df.groupby('Model', observed=False)['Avg_Score'].idxmax()]
        f.write("## Prompt hiệu quả nhất theo từng Model (dựa trên Avg_Score)\n\n")
        # MODIFIED: Display 'Avg_Score' in the table
        f.write(best_prompt_by_model[['Model', 'Prompt Strategy', 'Avg_Score', 'Generation Time (s)']].to_markdown())
        f.write("\n\n")

        # --- MODIFIED SECTION ---
        # Add the prompt strategy table to the report
        avg_by_prompt = df.groupby('Prompt Strategy').mean(numeric_only=True)
        prompt_counts = df.groupby('Prompt Strategy').size().rename('Usage Count')
        prompt_analysis = pd.concat([avg_by_prompt, prompt_counts], axis=1)
        # MODIFIED: Sort by 'Avg_Score'
        prompt_analysis = prompt_analysis.sort_values(by='Avg_Score', ascending=False)
        f.write("## Hiệu suất trung bình theo Prompt Strategy (Bảng)\n\n")
        f.write(prompt_analysis.to_markdown())
        f.write("\n\n")
        # --- END MODIFIED SECTION ---

        f.write("## Biểu đồ trực quan\n\n")

        # Conditionally add links to graphs
        if GENERATE_BAR_CHART:
            # MODIFIED: Update title and filename
            f.write(f"### Điểm Average Score theo Model và Prompt\n")
            graph1_basename = os.path.basename('avg_score_barplot.png')
            f.write(f"![Biểu đồ cột điểm Average Score]({graph1_basename})\n\n")

        if GENERATE_SCATTER_PLOT:
            # MODIFIED: Update title and filename
            f.write(f"### Generation time và điểm Average Score (có đường hồi quy)\n")
            graph2_basename = os.path.basename('generation_time_vs_avg_score_regression.png')
            f.write(f"![Biểu đồ phân tán thời gian và điểm Average Score]({graph2_basename})\n\n")
        
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

    # Bar chart of avg_score by Model & Prompt Strategy
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(
        data=df_with_gaps,
        x='Avg_Score',
        y='Model & Prompt Strategy',
        hue='Model',
        dodge=False,
        palette='viridis'
    )
    plt.title('Điểm Average Score theo Model và Prompt')
    plt.xlabel('Điểm Average Score')
    plt.ylabel('Model & Prompt Strategy')
    # plt.legend(title='Model', bbox_to_anchor=(0, 0), loc='upper left')
    ax.get_legend().remove()
    plt.tight_layout()

    lowest_prompts_per_model_full_string = df.loc[df.groupby('Model', observed=False)['Avg_Score'].idxmin()]['Model & Prompt Strategy'].tolist()

    for label_text in ax.get_yticklabels():
        full_label_text = label_text.get_text()
        start_index = full_label_text.find('(')
        end_index = full_label_text.find(')')
        if start_index != -1 and end_index != -1:
            prompt_strategy = full_label_text[start_index + 1 : end_index]
            if prompt_strategy == 'Extract':
                label_text.set_color('darkgreen')
            elif prompt_strategy == 'Current_Best':
                label_text.set_color('blue')
            elif full_label_text in lowest_prompts_per_model_full_string:
                label_text.set_color('red')

    graph1_filename = os.path.join(results_dir, 'average_score_barplot.png')
    plt.savefig(graph1_filename)
    plt.close()
    print(f"Biểu đồ '{graph1_filename}' đã được lưu.")

# --- FEATURE 3: GENERATE SCATTER PLOT ---
if GENERATE_SCATTER_PLOT:
    # --- Step 1: Add logic to identify outlier indices first ---
    all_outlier_indices = []
    actual_models_in_df = df['Model'].dropna().unique().tolist()
    for model_name in actual_models_in_df:
        model_df = df[df['Model'] == model_name]
        
        # An outlier calculation is only meaningful with several data points.
        if len(model_df) < 5:
            continue
            
        # Calculate IQR (Interquartile Range) for Avg_Score
        Q1 = model_df['Avg_Score'].quantile(0.25)
        Q3 = model_df['Avg_Score'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find outlier indices based on Avg_Score
        outlier_mask = (model_df['Avg_Score'] < lower_bound) | (model_df['Avg_Score'] > upper_bound)
        outlier_indices_for_model = model_df[outlier_mask].index
        all_outlier_indices.extend(outlier_indices_for_model)

    # --- Graph 1: Your original plot, now with outlier highlighting activated ---
    # Scatter plot of Generation Time vs. Avg_Score with Regression Lines
    plt.figure(figsize=(10, 6))
    
    ax = sns.scatterplot(
        data=df, 
        x='Generation Time (s)', 
        y='Avg_Score', 
        hue='Model', 
        style='Prompt Strategy', 
        s=100,
    )
    
    handles, labels = ax.get_legend_handles_labels()
    
    model_colors = {}
    for i, label in enumerate(labels):
        if label in df['Model'].dropna().unique():
            if hasattr(handles[i], 'get_facecolor') and len(handles[i].get_facecolor()) > 0:
                 model_colors[label] = handles[i].get_facecolor()[0]
            elif hasattr(handles[i], 'get_color'):
                 model_colors[label] = handles[i].get_color()
    
    if not model_colors:
        unique_models = df['Model'].dropna().unique()
        default_palette = sns.color_palette(n_colors=len(unique_models))
        model_colors = dict(zip(unique_models, default_palette))

    for model_name in df['Model'].dropna().unique():
        model_df = df[df['Model'] == model_name]
        if len(model_df) > 1:
            sns.regplot(
                data=model_df,
                x='Generation Time (s)',
                y='Avg_Score',
                scatter=False,
                ci=None,
                ax=ax,
                color=model_colors.get(model_name),
                line_kws={'alpha': 0.1}
            )
    
    # # --- Highlight Statistical Outliers on the first plot ---
    # if all_outlier_indices:
    #     outliers_df = df.loc[all_outlier_indices]
    #     ax.scatter(
    #         outliers_df['Generation Time (s)'],
    #         outliers_df['Avg_Score'],
    #         s=200,
    #         facecolors='none',
    #         edgecolors='red', # Changed to red for better visibility
    #         linewidths=1.5,
    #         label='_nolegend_'
    #     )
    # # --- End of Outlier Highlighting Section ---

    for model_name in df['Model'].dropna().unique():
        model_df = df[df['Model'] == model_name].copy()
        if len(model_df) <= 1:
            continue
        best_prompt_row = model_df.loc[model_df['Avg_Score'].idxmax()]
        worst_prompt_row = model_df.loc[model_df['Avg_Score'].idxmin()]
        best_label = best_prompt_row['Prompt Strategy']
        worst_label = worst_prompt_row['Prompt Strategy']
        model_color = model_colors.get(model_name)
        font_size = 9
        ax.text(
            x=best_prompt_row['Generation Time (s)'],
            y=best_prompt_row['Avg_Score'],
            s='    {}'.format(best_label),
            fontdict={'size': font_size, 'color': model_color, 'weight': 'bold'},
            ha='left',
            va='center'
        )
        ax.text(
            x=worst_prompt_row['Generation Time (s)'],
            y=worst_prompt_row['Avg_Score'],
            s='    {}'.format(worst_label),
            fontdict={'size': font_size, 'color': model_color},
            ha='left',
            va='center'
        )

    plt.title('Generation time & Avg_Score')
    plt.xlabel('Generation time (s)')
    plt.ylabel('Avg_Score')
    legend = ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 1.005, 0.975])
    
    fig = plt.gcf()
    renderer = fig.canvas.get_renderer() 
    legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    note_text = "Best strategy per model is bold"
    note_x = legend_bbox.x0
    note_y = legend_bbox.y0 - 0.02 # Adjusted y-offset for two lines
    plt.figtext(
        x=note_x, y=note_y, s=note_text, ha='left', fontsize=9,
        color='black', weight='bold', va='top'
    )
    
    plt.tight_layout()
    # Use a new filename for the first graph
    graph2_filename = os.path.join(results_dir, 'generation_time_vs_avg_score_with_outliers.png')
    plt.savefig(graph2_filename)
    plt.close()
    print(f"Biểu đồ '{graph2_filename}' đã được lưu.")

    # --- Step 2: Add logic to create the second graph without outliers ---
    df_no_outliers = df.drop(index=all_outlier_indices).reset_index(drop=True)

    # --- Graph 2: Your plotting code duplicated, using the filtered DataFrame ---
    plt.figure(figsize=(10, 6))
    
    # Use df_no_outliers as the data source
    ax = sns.scatterplot(
        data=df_no_outliers, 
        x='Generation Time (s)', 
        y='Avg_Score', 
        hue='Model', 
        style='Prompt Strategy', 
        s=100,
    )
    
    handles, labels = ax.get_legend_handles_labels()
    
    # Note: model_colors dictionary is reused from before, which is fine.
    
    # # Regression lines on filtered data
    # for model_name in df_no_outliers['Model'].dropna().unique():
    #     model_df = df_no_outliers[df_no_outliers['Model'] == model_name]
    #     if len(model_df) > 1:
    #         sns.regplot(
    #             data=model_df,
    #             x='Generation Time (s)',
    #             y='Avg_Score',
    #             scatter=False,
    #             ci=None,
    #             ax=ax,
    #             color=model_colors.get(model_name),
    #             line_kws={'alpha': 0.1}
    #         )

    # Annotations on filtered data (best/worst may change)
    for model_name in df_no_outliers['Model'].dropna().unique():
        model_df = df_no_outliers[df_no_outliers['Model'] == model_name].copy()
        if len(model_df) <= 1:
            continue
        best_prompt_row = model_df.loc[model_df['Avg_Score'].idxmax()]
        worst_prompt_row = model_df.loc[model_df['Avg_Score'].idxmin()]
        best_label = best_prompt_row['Prompt Strategy']
        worst_label = worst_prompt_row['Prompt Strategy']
        model_color = model_colors.get(model_name)
        font_size = 9
        ax.text(
            x=best_prompt_row['Generation Time (s)'],
            y=best_prompt_row['Avg_Score'],
            s='    {}'.format(best_label),
            fontdict={'size': font_size, 'color': model_color, 'weight': 'bold'},
            ha='left',
            va='center'
        )
        ax.text(
            x=worst_prompt_row['Generation Time (s)'],
            y=worst_prompt_row['Avg_Score'],
            s='    {}'.format(worst_label),
            fontdict={'size': font_size, 'color': model_color},
            ha='left',
            va='center'
        )

    plt.title('Generation time & Avg_Score (Outliers Removed)') # New title
    plt.xlabel('Generation time (s)')
    plt.ylabel('Avg_Score')
    legend = ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 1.005, 0.975])
    
    fig = plt.gcf()
    renderer = fig.canvas.get_renderer() 
    legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    note_text = "Best strategy per model is bold" # Original note
    note_x = legend_bbox.x0
    note_y = legend_bbox.y0 - 0.03
    plt.figtext(
        x=note_x, y=note_y, s=note_text, ha='left', fontsize=9,
        color='black', weight='bold'
    )
    
    plt.tight_layout()
    # Use a new filename for the second graph
    graph3_filename = os.path.join(results_dir, 'generation_time_vs_avg_score_no_outliers.png')
    plt.savefig(graph3_filename)
    plt.close()
    print(f"Biểu đồ '{graph3_filename}' đã được lưu.")
    
    # --- FEATURE 4: ANALYZE BY PROMPT STRATEGY ---
if GENERATE_PROMPT_ANALYSIS_GRAPH:
    # Calculate average performance metrics for each prompt strategy
    avg_by_prompt = df.groupby('Prompt Strategy').mean(numeric_only=True)
    # Sort by Avg_Score to have the best-performing prompts at the top
    prompt_analysis_sorted = avg_by_prompt.sort_values(by='Avg_Score', ascending=False)

    # Create the bar chart
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        data=prompt_analysis_sorted,
        x='Avg_Score',
        y=prompt_analysis_sorted.index, # Use the index (Prompt Strategy names) for the y-axis
        hue='Avg_Score',
        dodge=False,
        palette='plasma'
    )
    plt.title('Hiệu suất trung bình theo Prompt Strategy')
    plt.xlabel('Điểm Avg_Score trung bình')
    plt.ylabel('Prompt Strategy')
    ax.get_legend().remove()
    plt.tight_layout()

    # Save the graph to a file
    graph3_filename = os.path.join(results_dir, 'prompt_strategy_performance.png')
    plt.savefig(graph3_filename)
    plt.close()
    print(f"Biểu đồ phân tích Prompt Strategy đã được lưu tại: '{graph3_filename}'")