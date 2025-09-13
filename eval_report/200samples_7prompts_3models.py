import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

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

# --- Create a new directory for results ---
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
script_name_base = os.path.splitext(os.path.basename(script_path))[0]
results_dir = os.path.join(script_dir, script_name_base)
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved in: {results_dir}")

# --- Save Summary Tables to a Markdown file within the new directory ---
report_filename = os.path.join(results_dir, 'performance_report.md') # Changed extension to .md
with open(report_filename, 'w') as f:
    f.write("# Language Model Performance Report\n\n")

    # Average performance by model
    avg_performance_by_model = df.groupby('Model').mean(numeric_only=True)
    f.write("## Average Performance by Model\n\n")
    f.write(avg_performance_by_model.to_markdown()) # <--- Changed to .to_markdown()
    f.write("\n\n") # Add extra newlines for proper Markdown spacing

    # Best performing prompt strategy for each model based on BERTScore-F1
    best_prompt_by_model = df.loc[df.groupby('Model')['BERTScore-F1'].idxmax()]
    f.write("## Best Performing Prompt Strategy by Model (based on BERTScore-F1)\n\n")
    f.write(best_prompt_by_model[['Model', 'Prompt Strategy', 'BERTScore-F1', 'Generation Time (s)']].to_markdown()) # <--- Changed to .to_markdown()
    f.write("\n\n") # Add extra newlines for proper Markdown spacing

    f.write("## Linear Regression Analysis (BERTScore-F1 vs. Generation Time)\n\n")
    for model_name in df['Model'].unique():
        model_df = df[df['Model'] == model_name]
        # Skip models with fewer than 2 data points for regression, as it would be ill-defined.
        if len(model_df) < 2:
            f.write(f"\n### Model: {model_name.strip()}\n")
            f.write(f"  *Not enough data points ({len(model_df)}) for meaningful linear regression.*\n")
            continue

        X = model_df[['Generation Time (s)']]
        y = model_df['BERTScore-F1']

        from sklearn.linear_model import LinearRegression # Moved import here to keep main imports clean
        reg = LinearRegression().fit(X, y)
        r_squared = reg.score(X, y)
        coef = reg.coef_[0]
        intercept = reg.intercept_

        f.write(f"\n### Model: {model_name.strip()}\n")
        f.write(f"```\n") # Markdown code block for better readability of results
        f.write(f"R-squared: {r_squared:.4f}\n")
        f.write(f"Coefficient (slope): {coef:.8f}\n")
        f.write(f"Intercept: {intercept:.4f}\n")
        f.write(f"Interpretation: For every 1-second increase in generation time, the BERTScore-F1 is predicted to decrease by {abs(coef):.8f}.\n")
        f.write(f"```\n") # End of code block
    f.write("\n") # Ensure a newline at the end

    # Add links to the saved graphs
    f.write("## Visualizations\n\n")
    f.write(f"### BERTScore-F1 Scores by Model and Prompt Strategy\n")
    f.write(f"![BERTScore-F1 Scores Bar Plot]({os.path.basename(os.path.join(results_dir, 'bertscore_f1_scores_barplot.png'))})\n\n")
    f.write(f"### Generation Time vs. BERTScore-F1 with Linear Regression\n")
    f.write(f"![Generation Time vs. BERTScore-F1 Scatter Plot]({os.path.basename(os.path.join(results_dir, 'generation_time_vs_bertscore_regression.png'))})\n\n")


print(f"Textual report saved to {report_filename}")

# --- Generate and Save Graphs within the new directory ---
# Bar chart of BERTScore-F1 scores by Model and Prompt Strategy
plt.figure(figsize=(12, 8))
sns.barplot(
    data=df,
    x='BERTScore-F1',
    y='Model & Prompt Strategy',
    hue='Model & Prompt Strategy',
    palette='viridis',
    legend=False
)
plt.title('BERTScore-F1 Scores by Model and Prompt Strategy')
plt.xlabel('BERTScore-F1 Score')
plt.ylabel('Model & Prompt Strategy')
plt.tight_layout()
graph1_filename = os.path.join(results_dir, 'bertscore_f1_scores_barplot.png')
plt.savefig(graph1_filename)
plt.close()
print(f"Graph '{graph1_filename}' saved.")


# Scatter plot of Generation Time vs. BERTScore-F1 with Regression Lines
plt.figure(figsize=(10, 6))
ax = sns.scatterplot(data=df, x='Generation Time (s)', y='BERTScore-F1', hue='Model', style='Prompt Strategy', s=100)

for model_name in df['Model'].unique():
    model_df = df[df['Model'] == model_name]
    # Ensure there's enough data for regression line calculation
    if len(model_df) > 1: # Regression needs at least 2 points
        sns.regplot(
            data=model_df,
            x='Generation Time (s)',
            y='BERTScore-F1',
            scatter=False,
            ci=None,
            ax=ax,
            line_kws={'alpha': 0.2}
        )

plt.title('Generation Time vs. BERTScore-F1 with Linear Regression')
plt.xlabel('Generation Time (s)')
plt.ylabel('BERTScore-F1')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
graph2_filename = os.path.join(results_dir, 'generation_time_vs_bertscore_regression.png')
plt.savefig(graph2_filename)
plt.close()
print(f"Graph '{graph2_filename}' saved.")