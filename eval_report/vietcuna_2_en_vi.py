import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import os
import sys

# 1. Load the data from the string format provided
# The data is copied directly from the user's prompt
data = """
Model	Prompt_Strategy	System_Prompt	ROUGE-L	BLEU	METEOR	BERTScore-F1	Avg-Score	Generation Time (s)
0	vilm/vietcuna-3b-v2	Extract_VI	Empty_SP_VI	0.5516	0.2677	0.4841	0.8425	0.5365	3.53
1	vilm/vietcuna-3b-v2	List_EN	Expert_SP_EN	0.4921	0.2148	0.6086	0.7969	0.5281	7.79
2	vilm/vietcuna-3b-v2	Concise_EN	Default_SP_EN	0.4485	0.2692	0.5883	0.8025	0.5271	6.78
3	vilm/vietcuna-3b-v2	Direct_VI	Default_SP_VI	0.5339	0.2476	0.4954	0.8286	0.5264	4.74
4	vilm/vietcuna-3b-v2	Direct_EN	Empty_SP_EN	0.3912	0.3099	0.4904	0.8032	0.4987	4.18
5	vilm/vietcuna-3b-v2	Extract_EN	Default_SP_EN	0.3995	0.3004	0.4894	0.8017	0.4978	4.43
6	vilm/vietcuna-3b-v2	Direct_EN	Default_SP_EN	0.4820	0.3323	0.3811	0.7804	0.4940	4.89
7	vilm/vietcuna-3b-v2	List_EN	Empty_SP_EN	0.3965	0.2248	0.5095	0.8061	0.4842	4.34
8	vilm/vietcuna-3b-v2	Concise_EN	Empty_SP_EN	0.3938	0.2215	0.5090	0.8046	0.4822	4.41
9	vilm/vietcuna-3b-v2	Extract_EN	Empty_SP_EN	0.3938	0.2215	0.5090	0.8046	0.4822	4.36
10	vilm/vietcuna-3b-v2	Strict_Rules_EN	Empty_SP_EN	0.3633	0.2281	0.5231	0.8008	0.4788	4.91
11	vilm/vietcuna-3b-v2	Concise_VI	Empty_SP_VI	0.3844	0.1816	0.5224	0.7947	0.4708	5.72
12	vilm/vietcuna-3b-v2	List_EN	Default_SP_EN	0.3793	0.1941	0.5044	0.7836	0.4653	6.41
13	vilm/vietcuna-3b-v2	Strict_Rules_EN	Expert_SP_EN	0.3907	0.2218	0.4504	0.7893	0.4630	5.89
14	vilm/vietcuna-3b-v2	Direct_EN	Expert_SP_EN	0.3856	0.2169	0.4556	0.7906	0.4622	5.64
15	vilm/vietcuna-3b-v2	Extract_EN	Expert_SP_EN	0.3849	0.2121	0.4532	0.7777	0.4570	5.96
16	vilm/vietcuna-3b-v2	List_VI	Default_SP_VI	0.4882	0.2654	0.2664	0.7664	0.4466	4.49
17	vilm/vietcuna-3b-v2	Concise_VI	Default_SP_VI	0.4537	0.2560	0.2457	0.7621	0.4294	4.37
18	vilm/vietcuna-3b-v2	Concise_VI	Expert_SP_VI	0.2989	0.1190	0.5167	0.7560	0.4226	8.91
19	vilm/vietcuna-3b-v2	Extract_VI	Default_SP_VI	0.4423	0.2529	0.2312	0.7584	0.4212	4.39
20	vilm/vietcuna-3b-v2	List_VI	Empty_SP_VI	0.4205	0.1341	0.2751	0.7485	0.3946	5.81
21	vilm/vietcuna-3b-v2	Extract_VI	Expert_SP_VI	0.3723	0.1479	0.2622	0.7442	0.3816	5.60
22	vilm/vietcuna-3b-v2	Strict_Rules_VI	Empty_SP_VI	0.3297	0.0824	0.2899	0.7313	0.3583	8.96
23	vilm/vietcuna-3b-v2	Direct_VI	Empty_SP_VI	0.3452	0.0947	0.2710	0.7221	0.3582	7.75
24	vilm/vietcuna-3b-v2	Strict_Rules_VI	Default_SP_VI	0.3271	0.0959	0.2519	0.7010	0.3440	8.47
25	vilm/vietcuna-3b-v2	List_VI	Expert_SP_VI	0.3512	0.0511	0.2640	0.6928	0.3398	15.35
26	vilm/vietcuna-3b-v2	Direct_VI	Expert_SP_VI	0.2963	0.0392	0.2725	0.6811	0.3223	16.54
27	vilm/vietcuna-3b-v2	Concise_EN	Expert_SP_EN	0.3136	0.0534	0.2160	0.6813	0.3161	15.53
28	vilm/vietcuna-3b-v2	Strict_Rules_EN	Default_SP_EN	0.3029	0.0539	0.2223	0.6785	0.3144	15.50
29	vilm/vietcuna-3b-v2	Strict_Rules_VI	Expert_SP_VI	0.2216	0.0358	0.2440	0.6317	0.2833	20.07
"""

# Use io.StringIO to read the string data into a pandas DataFrame
df = pd.read_csv(io.StringIO(data), sep='\t')

# 2. Process the data to create comparable pairs
# Extract the base name for Prompt Strategy (e.g., 'Extract')
df['Strategy'] = df['Prompt_Strategy'].str.replace('_VI|_EN', '', regex=True)

# Extract the base name for System Prompt (e.g., 'Default')
df['System'] = df['System_Prompt'].str.replace('_SP_VI|_SP_EN', '', regex=True)

# Identify the language
df['Language'] = df['Prompt_Strategy'].apply(lambda x: 'Vietnamese' if '_VI' in x else 'English')

# Create a combined pair name for the x-axis labels
df['Pair'] = df['Strategy'] + ' / ' + df['System']

# Group by the 'Language' column and calculate the mean of the 'Avg-Score'
average_scores = df.groupby('Language')['Avg-Score'].mean()

print("--- Overall Average Score Comparison ---")
print(average_scores)
print("----------------------------------------\n")

# 3. Pivot the table to get Vietnamese and English scores in separate columns
pivot_df = df.pivot_table(index='Pair', columns='Language', values='Avg-Score')

# Sort the pairs to have a consistent order in the plot
pivot_df = pivot_df.sort_index()

# 4. Generate the bar chart
labels = pivot_df.index
vietnamese_scores = pivot_df['Vietnamese']
english_scores = pivot_df['English']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, vietnamese_scores, width, label='Vietnamese', color='blue', alpha=0.8)
rects2 = ax.bar(x + width/2, english_scores, width, label='English', color='green', alpha=0.8)

# Add some text for labels, title and axes ticks
ax.set_ylabel('Average Score')
ax.set_xlabel('Prompt Strategy / System Prompt Pair')
ax.set_title('Comparison of Vietnamese vs. English Prompt Performance')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()

# Improve layout to prevent labels from overlapping
fig.tight_layout()

# 5. Save the graph to a file
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
script_name_base = os.path.splitext(os.path.basename(script_path))[0]
results_dir = os.path.join(script_dir, script_name_base)
os.makedirs(results_dir, exist_ok=True)
output_filename = os.path.join(results_dir, 'vietcuna_2_en_vi.png')
plt.savefig(output_filename, dpi=300) # dpi for higher resolution

print(f"Graph saved successfully as '{output_filename}'")