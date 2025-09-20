import pandas as pd
import io

# --- Data from each seed stored as multiline strings ---

seed_42_data = """
|     | Model                 | Prompt_Strategy | System_Prompt | ROUGE-L | BLEU   | METEOR | BERTScore-F1 | Avg-Score | Generation Time (s) |
| --- | --------------------- | --------------- | ------------- | ------- | ------ | ------ | ------------ | --------- | ------------------- |
| 0   | arcee-ai/Arcee-VyLinh | Concise_EN      | Expert_SP_EN  | 0.6663  | 0.4055 | 0.6387 | 0.8781       | 0.6472    | 60.58               |
| 1   | arcee-ai/Arcee-VyLinh | Extract_VI      | Expert_SP_VI  | 0.6493  | 0.4185 | 0.6277 | 0.8472       | 0.6357    | 66.05               |
| 2   | arcee-ai/Arcee-VyLinh | Extract_EN      | Expert_SP_EN  | 0.6299  | 0.3518 | 0.5980 | 0.8574       | 0.6093    | 53.78               |
| 3   | arcee-ai/Arcee-VyLinh | Strict_Rules_VI | Expert_SP_VI  | 0.5927  | 0.4207 | 0.5749 | 0.8424       | 0.6077    | 71.89               |
| 4   | arcee-ai/Arcee-VyLinh | Concise_VI      | Expert_SP_VI  | 0.5588  | 0.3269 | 0.4940 | 0.8311       | 0.5527    | 61.09               |
| 5   | arcee-ai/Arcee-VyLinh | Strict_Rules_EN | Expert_SP_EN  | 0.5424  | 0.3469 | 0.4885 | 0.8196       | 0.5494    | 57.69               |
"""

seed_1_data = """
|     | Model                 | Prompt_Strategy | System_Prompt | ROUGE-L | BLEU   | METEOR | BERTScore-F1 | Avg-Score | Generation Time (s) |
| --- | --------------------- | --------------- | ------------- | ------- | ------ | ------ | ------------ | --------- | ------------------- |
| 0   | arcee-ai/Arcee-VyLinh | Extract_EN      | Expert_SP_EN  | 0.6835  | 0.4708 | 0.6354 | 0.8831       | 0.6682    | 65.88               |
| 1   | arcee-ai/Arcee-VyLinh | Strict_Rules_EN | Expert_SP_EN  | 0.6424  | 0.4580 | 0.5676 | 0.8690       | 0.6342    | 69.95               |
| 2   | arcee-ai/Arcee-VyLinh | Concise_EN      | Expert_SP_EN  | 0.6142  | 0.4790 | 0.5771 | 0.8629       | 0.6333    | 73.10               |
| 3   | arcee-ai/Arcee-VyLinh | Strict_Rules_VI | Expert_SP_VI  | 0.6239  | 0.4189 | 0.5686 | 0.8567       | 0.6170    | 70.28               |
| 4   | arcee-ai/Arcee-VyLinh | Extract_VI      | Expert_SP_VI  | 0.5651  | 0.4657 | 0.5488 | 0.8510       | 0.6076    | 70.28               |
| 5   | arcee-ai/Arcee-VyLinh | Concise_VI      | Expert_SP_VI  | 0.5119  | 0.3798 | 0.4639 | 0.8141       | 0.5424    | 71.64               |
"""

seed_2_data = """
|     | Model                 | Prompt_Strategy | System_Prompt | ROUGE-L | BLEU   | METEOR | BERTScore-F1 | Avg-Score | Generation Time (s) |
| --- | --------------------- | --------------- | ------------- | ------- | ------ | ------ | ------------ | --------- | ------------------- |
| 0   | arcee-ai/Arcee-VyLinh | Extract_EN      | Expert_SP_EN  | 0.5952  | 0.3963 | 0.5823 | 0.8596       | 0.6083    | 59.94               |
| 1   | arcee-ai/Arcee-VyLinh | Concise_EN      | Expert_SP_EN  | 0.5259  | 0.3368 | 0.4779 | 0.8211       | 0.5404    | 64.93               |
| 2   | arcee-ai/Arcee-VyLinh | Strict_Rules_VI | Expert_SP_VI  | 0.4857  | 0.2896 | 0.4483 | 0.8010       | 0.5062    | 63.67               |
| 3   | arcee-ai/Arcee-VyLinh | Extract_VI      | Expert_SP_VI  | 0.4863  | 0.2885 | 0.4347 | 0.8069       | 0.5041    | 62.44               |
| 4   | arcee-ai/Arcee-VyLinh | Strict_Rules_EN | Expert_SP_EN  | 0.4522  | 0.2670 | 0.4087 | 0.7941       | 0.4805    | 58.21               |
| 5   | arcee-ai/Arcee-VyLinh | Concise_VI      | Expert_SP_VI  | 0.4058  | 0.1585 | 0.3357 | 0.7662       | 0.4166    | 62.61               |
"""

seed_3_data = """
|     | Model                 | Prompt_Strategy | System_Prompt | ROUGE-L | BLEU   | METEOR | BERTScore-F1 | Avg-Score | Generation Time (s) |
| --- | --------------------- | --------------- | ------------- | ------- | ------ | ------ | ------------ | --------- | ------------------- |
| 0   | arcee-ai/Arcee-VyLinh | Extract_VI      | Expert_SP_VI  | 0.5913  | 0.5434 | 0.5691 | 0.8406       | 0.6361    | 71.19               |
| 1   | arcee-ai/Arcee-VyLinh | Strict_Rules_VI | Expert_SP_VI  | 0.5573  | 0.5258 | 0.5746 | 0.8287       | 0.6216    | 83.08               |
| 2   | arcee-ai/Arcee-VyLinh | Extract_EN      | Expert_SP_EN  | 0.5463  | 0.4673 | 0.5588 | 0.8355       | 0.6020    | 67.00               |
| 3   | arcee-ai/Arcee-VyLinh | Concise_EN      | Expert_SP_EN  | 0.5035  | 0.4324 | 0.5197 | 0.8164       | 0.5680    | 71.55               |
| 4   | arcee-ai/Arcee-VyLinh | Concise_VI      | Expert_SP_VI  | 0.4483  | 0.2403 | 0.3930 | 0.7882       | 0.4674    | 68.88               |
| 5   | arcee-ai/Arcee-VyLinh | Strict_Rules_EN | Expert_SP_EN  | 0.3991  | 0.2983 | 0.3931 | 0.7621       | 0.4632    | 67.88               |
"""

seed_4_data = """
|     | Model                 | Prompt_Strategy | System_Prompt | ROUGE-L | BLEU   | METEOR | BERTScore-F1 | Avg-Score | Generation Time (s) |
| --- | --------------------- | --------------- | ------------- | ------- | ------ | ------ | ------------ | --------- | ------------------- |
| 0   | arcee-ai/Arcee-VyLinh | Extract_EN      | Expert_SP_EN  | 0.5596  | 0.3800 | 0.5526 | 0.8352       | 0.5818    | 79.25               |
| 1   | arcee-ai/Arcee-VyLinh | Strict_Rules_EN | Expert_SP_EN  | 0.5420  | 0.3792 | 0.5271 | 0.8268       | 0.5688    | 82.62               |
| 2   | arcee-ai/Arcee-VyLinh | Concise_EN      | Expert_SP_EN  | 0.5331  | 0.3610 | 0.5184 | 0.8255       | 0.5595    | 85.15               |
| 3   | arcee-ai/Arcee-VyLinh | Extract_VI      | Expert_SP_VI  | 0.5285  | 0.3116 | 0.5182 | 0.8221       | 0.5451    | 87.21               |
| 4   | arcee-ai/Arcee-VyLinh | Strict_Rules_VI | Expert_SP_VI  | 0.4676  | 0.3092 | 0.4403 | 0.7924       | 0.5024    | 91.03               |
| 5   | arcee-ai/Arcee-VyLinh | Concise_VI      | Expert_SP_VI  | 0.4699  | 0.2203 | 0.3922 | 0.7866       | 0.4672    | 71.24               |
"""

def parse_markdown_table(markdown_string):
    """Parses a Markdown table string into a clean pandas DataFrame."""
    string_reader = io.StringIO(markdown_string.strip())
    df = pd.read_csv(string_reader, sep='|', index_col=1)
    df = df.drop(df.index[0])
    df = df.iloc[:, 1:-1]
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df

# 1. Parse all data and combine into a single DataFrame
all_seeds_data = [seed_42_data, seed_1_data, seed_2_data, seed_3_data, seed_4_data]
df_list = [parse_markdown_table(data) for data in all_seeds_data]
combined_df = pd.concat(df_list).reset_index(drop=True)

# 2. Identify numeric columns that need to be averaged
numeric_columns = ['ROUGE-L', 'BLEU', 'METEOR', 'BERTScore-F1', 'Avg-Score', 'Generation Time (s)']
# Convert these columns to a numeric type so we can perform calculations
for col in numeric_columns:
    combined_df[col] = pd.to_numeric(combined_df[col])

# 3. Group by the prompt pair and calculate the mean for numeric columns
# The 'as_index=False' argument keeps 'Prompt_Strategy' and 'System_Prompt' as columns
averaged_df = combined_df.groupby(['Prompt_Strategy', 'System_Prompt'], as_index=False)[numeric_columns].mean()

# 4. Sort the results by the 'Avg-Score' in descending order to see the best performers
averaged_df_sorted = averaged_df.sort_values(by='Avg-Score', ascending=False)


# 5. Display the final aggregated and sorted table
print("--- Averaged Performance Across All Seeds ---")
print(averaged_df_sorted.to_string())