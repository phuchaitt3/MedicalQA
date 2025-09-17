import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import sys

data = """
Model	Prompt_Strategy	System_Prompt	ROUGE-L	BLEU	METEOR	BERTScore-F1	Avg-Score	Generation Time (s)
0	arcee-ai/Arcee-VyLinh	Concise_VI	Expert_SP	0.7405	0.5868	0.6952	0.8934	0.7290	N/A
1	arcee-ai/Arcee-VyLinh	Extract_VI	Default_SP	0.7287	0.5374	0.7169	0.8914	0.7186	N/A
2	arcee-ai/Arcee-VyLinh	Extract_VI	Expert_SP	0.7254	0.5130	0.7033	0.8901	0.7080	N/A
3	arcee-ai/Arcee-VyLinh	Strict_Rules_VI	Expert_SP	0.7093	0.5157	0.6984	0.8737	0.6993	N/A
4	arcee-ai/Arcee-VyLinh	Strict_Rules_VI	Default_SP	0.7093	0.5157	0.6984	0.8737	0.6993	N/A
5	arcee-ai/Arcee-VyLinh	Strict_Rules_VI	Empty_SP	0.6847	0.5006	0.6373	0.8767	0.6748	N/A
6	arcee-ai/Arcee-VyLinh	Extract_VI	Empty_SP	0.6541	0.4847	0.6545	0.8766	0.6675	N/A
7	vilm/vinallama-2.7b-chat	Full_VI	Default_SP	0.6052	0.3090	0.7129	0.8692	0.6241	N/A
8	vilm/vietcuna-3b-v2	Strict_Rules_VI	Empty_SP	0.5685	0.2975	0.7127	0.8499	0.6072	N/A
9	vilm/vietcuna-3b-v2	List_VI	Expert_SP	0.5828	0.4057	0.5727	0.8577	0.6047	N/A
10	alpha-ai/LLAMA3-3B-Medical-COT	Direct_VI	Empty_SP	0.5940	0.3586	0.5986	0.8477	0.5997	N/A
11	arcee-ai/Arcee-VyLinh	Concise_VI	Default_SP	0.5569	0.4263	0.5505	0.8394	0.5933	N/A
12	vilm/vinallama-2.7b-chat	Full_VI	Expert_SP	0.5463	0.2700	0.6911	0.8545	0.5905	N/A
13	vilm/vietcuna-3b-v2	Concise_VI	Expert_SP	0.5335	0.3422	0.6217	0.8527	0.5875	N/A
14	vilm/vietcuna-3b-v2	Strict_Rules_VI	Default_SP	0.5413	0.2563	0.6998	0.8400	0.5843	N/A
15	vilm/vietcuna-3b-v2	List_VI	Default_SP	0.5404	0.3404	0.5796	0.8497	0.5775	N/A
16	vilm/vietcuna-3b-v2	Concise_VI	Default_SP	0.5404	0.3404	0.5796	0.8497	0.5775	N/A
17	vilm/vietcuna-3b-v2	Strict_Rules_VI	Expert_SP	0.5291	0.2424	0.6908	0.8398	0.5755	N/A
18	alpha-ai/LLAMA3-3B-Medical-COT	Full_VI	Empty_SP	0.5347	0.2359	0.6869	0.8294	0.5717	N/A
19	vilm/vietcuna-3b-v2	Full_VI	Empty_SP	0.5455	0.3508	0.5388	0.8423	0.5694	N/A
20	vilm/vietcuna-3b-v2	Self-Contained	ViMedAQA_SP	0.5390	0.3458	0.5339	0.8456	0.5661	N/A
21	vilm/vietcuna-3b-v2	Full_VI	Default_SP	0.5093	0.3218	0.5771	0.8430	0.5628	N/A
22	vilm/vietcuna-3b-v2	Concise_VI	Empty_SP	0.5116	0.3218	0.5704	0.8426	0.5616	N/A
23	vilm/vietcuna-3b-v2	Full_VI	Expert_SP	0.5373	0.3256	0.5369	0.8424	0.5606	N/A
24	arcee-ai/Arcee-VyLinh	Few_Shot_VI	Expert_SP	0.5153	0.2605	0.6521	0.8102	0.5595	N/A
25	vilm/vietcuna-3b-v2	List_VI	Empty_SP	0.5069	0.3092	0.5641	0.8431	0.5558	N/A
26	alpha-ai/LLAMA3-3B-Medical-COT	Strict_Rules_VI	Empty_SP	0.5790	0.3195	0.4606	0.8462	0.5513	N/A
27	arcee-ai/Arcee-VyLinh	Direct_VI	Default_SP	0.5281	0.2225	0.6276	0.8171	0.5488	N/A
28	alpha-ai/LLAMA3-3B-Medical-COT	Concise_VI	Expert_SP	0.5625	0.3154	0.4580	0.8487	0.5462	N/A
29	alpha-ai/LLAMA3-3B-Medical-COT	Self-Contained	ViMedAQA_SP	0.5497	0.2281	0.5758	0.8281	0.5454	N/A
30	alpha-ai/LLAMA3-3B-Medical-COT	Few_Shot_VI	Empty_SP	0.5181	0.3160	0.5207	0.8033	0.5395	N/A
31	vilm/vinallama-2.7b-chat	Few_Shot_VI	Empty_SP	0.5319	0.1549	0.6652	0.8055	0.5394	N/A
32	arcee-ai/Arcee-VyLinh	Concise_VI	Empty_SP	0.4557	0.4051	0.4753	0.7929	0.5322	N/A
33	alpha-ai/LLAMA3-3B-Medical-COT	Concise_VI	Empty_SP	0.5561	0.2621	0.4312	0.8430	0.5231	N/A
34	arcee-ai/Arcee-VyLinh	Direct_VI	Expert_SP	0.4669	0.1849	0.6462	0.7889	0.5217	N/A
35	arcee-ai/Arcee-VyLinh	Few_Shot_VI	Default_SP	0.4654	0.2003	0.6213	0.7958	0.5207	N/A
36	alpha-ai/LLAMA3-3B-Medical-COT	Few_Shot_VI	Default_SP	0.5008	0.2399	0.5069	0.7912	0.5097	N/A
37	alpha-ai/LLAMA3-3B-Medical-COT	Direct_VI	Default_SP	0.4782	0.1746	0.5844	0.7990	0.5090	N/A
38	arcee-ai/Arcee-VyLinh	ViMedAQA_VI	Default_SP	0.4535	0.1720	0.5949	0.8021	0.5056	N/A
39	alpha-ai/LLAMA3-3B-Medical-COT	Few_Shot_VI	Expert_SP	0.4734	0.2056	0.5399	0.7931	0.5030	N/A
40	arcee-ai/Arcee-VyLinh	ViMedAQA_VI	Expert_SP	0.4120	0.1638	0.6331	0.7977	0.5016	N/A
41	alpha-ai/LLAMA3-3B-Medical-COT	Direct_VI	Expert_SP	0.4896	0.1448	0.5514	0.7993	0.4963	N/A
42	vilm/vinallama-2.7b-chat	ViMedAQA_VI	Default_SP	0.4810	0.1400	0.5583	0.8040	0.4958	N/A
43	vilm/vinallama-2.7b-chat	ViMedAQA_VI	Expert_SP	0.4706	0.1524	0.5608	0.7994	0.4958	N/A
44	alpha-ai/LLAMA3-3B-Medical-COT	Strict_Rules_VI	Default_SP	0.5111	0.2240	0.4533	0.7887	0.4943	N/A
45	vilm/vietcuna-3b-v2	Direct_VI	Expert_SP	0.4650	0.1279	0.5614	0.8164	0.4927	N/A
46	alpha-ai/LLAMA3-3B-Medical-COT	ViMedAQA_VI	Expert_SP	0.4700	0.1543	0.5434	0.8016	0.4923	N/A
47	alpha-ai/LLAMA3-3B-Medical-COT	Extract_VI	Empty_SP	0.5001	0.2803	0.3814	0.8029	0.4912	N/A
48	vilm/vietcuna-3b-v2	ViMedAQA_VI	Default_SP	0.4783	0.1220	0.5308	0.8104	0.4854	N/A
49	vilm/vinallama-2.7b-chat	Few_Shot_VI	Expert_SP	0.4349	0.1237	0.5809	0.7935	0.4832	N/A
50	vilm/vietcuna-3b-v2	Extract_VI	Expert_SP	0.4822	0.1183	0.5263	0.8043	0.4828	N/A
51	alpha-ai/LLAMA3-3B-Medical-COT	Full_VI	Default_SP	0.4103	0.1544	0.5899	0.7707	0.4813	N/A
52	alpha-ai/LLAMA3-3B-Medical-COT	List_VI	Default_SP	0.4448	0.1777	0.4932	0.7879	0.4759	N/A
53	arcee-ai/Arcee-VyLinh	Self-Contained	ViMedAQA_SP	0.3780	0.1466	0.5963	0.7824	0.4758	N/A
54	vilm/vinallama-2.7b-chat	Few_Shot_VI	Default_SP	0.4138	0.1252	0.5635	0.7924	0.4737	N/A
55	vilm/vietcuna-3b-v2	Direct_VI	Default_SP	0.4550	0.1161	0.5180	0.7992	0.4721	N/A
56	vilm/vietcuna-3b-v2	Extract_VI	Default_SP	0.4217	0.1165	0.5399	0.7989	0.4693	N/A
57	vilm/vietcuna-3b-v2	Extract_VI	Empty_SP	0.4447	0.1170	0.5123	0.8003	0.4686	N/A
58	alpha-ai/LLAMA3-3B-Medical-COT	Extract_VI	Default_SP	0.4873	0.2288	0.3412	0.8008	0.4645	N/A
59	arcee-ai/Arcee-VyLinh	Direct_VI	Empty_SP	0.3409	0.1371	0.6080	0.7700	0.4640	N/A
60	alpha-ai/LLAMA3-3B-Medical-COT	ViMedAQA_VI	Empty_SP	0.4403	0.1329	0.5083	0.7741	0.4639	N/A
61	vilm/vinallama-2.7b-chat	Direct_VI	Default_SP	0.4495	0.0978	0.5263	0.7813	0.4637	N/A
62	alpha-ai/LLAMA3-3B-Medical-COT	Extract_VI	Expert_SP	0.4818	0.2304	0.3397	0.8008	0.4632	N/A
63	alpha-ai/LLAMA3-3B-Medical-COT	Strict_Rules_VI	Expert_SP	0.4668	0.1790	0.4190	0.7868	0.4629	N/A
64	vilm/vietcuna-3b-v2	Direct_VI	Empty_SP	0.4390	0.1064	0.5005	0.7899	0.4590	N/A
65	vilm/vietcuna-3b-v2	ViMedAQA_VI	Expert_SP	0.4333	0.1094	0.4946	0.7938	0.4578	N/A
66	arcee-ai/Arcee-VyLinh	List_VI	Expert_SP	0.5122	0.1833	0.3756	0.7401	0.4528	N/A
67	vilm/vinallama-2.7b-chat	Strict_Rules_VI	Default_SP	0.3779	0.1191	0.5329	0.7739	0.4510	N/A
68	arcee-ai/Arcee-VyLinh	Full_VI	Default_SP	0.3487	0.1046	0.5787	0.7670	0.4498	N/A
69	alpha-ai/LLAMA3-3B-Medical-COT	ViMedAQA_VI	Default_SP	0.4207	0.1167	0.4728	0.7709	0.4453	N/A
70	alpha-ai/LLAMA3-3B-Medical-COT	List_VI	Empty_SP	0.4248	0.1495	0.4499	0.7562	0.4451	N/A
71	arcee-ai/Arcee-VyLinh	Few_Shot_VI	Empty_SP	0.3197	0.1090	0.5870	0.7535	0.4423	N/A
72	arcee-ai/Arcee-VyLinh	List_VI	Default_SP	0.4459	0.2200	0.3642	0.7381	0.4420	N/A
73	arcee-ai/Arcee-VyLinh	ViMedAQA_VI	Empty_SP	0.3075	0.0974	0.5614	0.7639	0.4326	N/A
74	vilm/vinallama-2.7b-chat	Extract_VI	Expert_SP	0.3851	0.0836	0.5115	0.7419	0.4305	N/A
75	vilm/vietcuna-3b-v2	ViMedAQA_VI	Empty_SP	0.4021	0.0924	0.4436	0.7732	0.4278	N/A
76	arcee-ai/Arcee-VyLinh	List_VI	Empty_SP	0.3328	0.1258	0.4943	0.7227	0.4189	N/A
77	arcee-ai/Arcee-VyLinh	Full_VI	Empty_SP	0.3046	0.0872	0.5349	0.7439	0.4176	N/A
78	alpha-ai/LLAMA3-3B-Medical-COT	Full_VI	Expert_SP	0.3098	0.1038	0.5042	0.7462	0.4160	N/A
79	vilm/vinallama-2.7b-chat	Extract_VI	Default_SP	0.3364	0.0829	0.5197	0.7213	0.4151	N/A
80	alpha-ai/LLAMA3-3B-Medical-COT	Concise_VI	Default_SP	0.3959	0.1900	0.2663	0.7912	0.4108	N/A
81	alpha-ai/LLAMA3-3B-Medical-COT	List_VI	Expert_SP	0.4333	0.0797	0.3878	0.7333	0.4085	N/A
82	sail/Sailor-4B	Strict_Rules_VI	Empty_SP	0.2981	0.0855	0.5133	0.7035	0.4001	N/A
83	vilm/vietcuna-3b-v2	Few_Shot_VI	Expert_SP	0.3024	0.0683	0.4472	0.7118	0.3824	N/A
84	vilm/vietcuna-3b-v2	Few_Shot_VI	Default_SP	0.3069	0.0690	0.4425	0.7058	0.3810	N/A
85	arcee-ai/Arcee-VyLinh	Full_VI	Expert_SP	0.2396	0.0740	0.4735	0.7312	0.3796	N/A
86	sail/Sailor-4B	Direct_VI	Expert_SP	0.2286	0.0843	0.4984	0.6898	0.3753	N/A
87	vilm/vietcuna-3b-v2	Few_Shot_VI	Empty_SP	0.2871	0.0613	0.4145	0.6862	0.3623	N/A
88	sail/Sailor-4B	Strict_Rules_VI	Expert_SP	0.2412	0.0683	0.4649	0.6705	0.3612	N/A
89	sail/Sailor-4B	Strict_Rules_VI	Default_SP	0.2431	0.0632	0.4536	0.6740	0.3585	N/A
90	vilm/vinallama-2.7b-chat	Direct_VI	Expert_SP	0.2755	0.0605	0.4030	0.6871	0.3565	N/A
91	vilm/vinallama-2.7b-chat	Strict_Rules_VI	Expert_SP	0.2470	0.0669	0.4419	0.6606	0.3541	N/A
92	vilm/vinallama-2.7b-chat	Concise_VI	Empty_SP	0.2428	0.0707	0.3951	0.6689	0.3444	N/A
93	sail/Sailor-4B	Concise_VI	Expert_SP	0.2240	0.0710	0.4084	0.6555	0.3397	N/A
94	sail/Sailor-4B	Self-Contained	ViMedAQA_SP	0.2139	0.0653	0.4297	0.6491	0.3395	N/A
95	vilm/vinallama-2.7b-chat	Strict_Rules_VI	Empty_SP	0.1875	0.0627	0.4366	0.6664	0.3383	N/A
96	sail/Sailor-4B	Extract_VI	Default_SP	0.1784	0.0660	0.4159	0.6562	0.3291	N/A
97	sail/Sailor-4B	List_VI	Empty_SP	0.1867	0.0645	0.4243	0.6376	0.3283	N/A
98	sail/Sailor-4B	Full_VI	Expert_SP	0.1703	0.0632	0.4131	0.6625	0.3273	N/A
99	sail/Sailor-4B	ViMedAQA_VI	Expert_SP	0.2166	0.0633	0.3784	0.6447	0.3258	N/A
100	vilm/vinallama-2.7b-chat	Direct_VI	Empty_SP	0.1663	0.0609	0.4194	0.6550	0.3254	N/A
101	sail/Sailor-4B	Extract_VI	Expert_SP	0.1698	0.0636	0.4233	0.6438	0.3251	N/A
102	vilm/vinallama-2.7b-chat	List_VI	Empty_SP	0.1650	0.0600	0.4210	0.6530	0.3248	N/A
103	sail/Sailor-4B	Direct_VI	Empty_SP	0.1920	0.0630	0.3921	0.6496	0.3242	N/A
104	sail/Sailor-4B	Full_VI	Empty_SP	0.1706	0.0624	0.4164	0.6313	0.3202	N/A
105	sail/Sailor-4B	Few_Shot_VI	Expert_SP	0.1739	0.0615	0.4064	0.6376	0.3198	N/A
106	sail/Sailor-4B	List_VI	Default_SP	0.1784	0.0613	0.4095	0.6271	0.3191	N/A
107	vilm/vinallama-2.7b-chat	Full_VI	Empty_SP	0.1698	0.0577	0.3942	0.6510	0.3182	N/A
108	vilm/vinallama-2.7b-chat	Self-Contained	ViMedAQA_SP	0.1694	0.0555	0.4042	0.6412	0.3176	N/A
109	vilm/vinallama-2.7b-chat	Concise_VI	Expert_SP	0.1975	0.0533	0.3953	0.6223	0.3171	N/A
110	sail/Sailor-4B	Concise_VI	Empty_SP	0.1798	0.0595	0.3859	0.6408	0.3165	N/A
111	vilm/vinallama-2.7b-chat	List_VI	Expert_SP	0.1684	0.0544	0.3973	0.6418	0.3155	N/A
112	vilm/vinallama-2.7b-chat	List_VI	Default_SP	0.1622	0.0569	0.3900	0.6423	0.3128	N/A
113	sail/Sailor-4B	Direct_VI	Default_SP	0.1675	0.0563	0.4019	0.6190	0.3112	N/A
114	sail/Sailor-4B	Few_Shot_VI	Empty_SP	0.1643	0.0555	0.3630	0.6535	0.3091	N/A
115	sail/Sailor-4B	Extract_VI	Empty_SP	0.1620	0.0561	0.3796	0.6301	0.3070	N/A
116	sail/Sailor-4B	List_VI	Expert_SP	0.1877	0.0591	0.3402	0.6411	0.3070	N/A
117	vilm/vinallama-2.7b-chat	Concise_VI	Default_SP	0.1976	0.0534	0.3701	0.6039	0.3062	N/A
118	sail/Sailor-4B	Full_VI	Default_SP	0.1642	0.0581	0.3754	0.6255	0.3058	N/A
119	sail/Sailor-4B	Few_Shot_VI	Default_SP	0.1610	0.0535	0.3581	0.6408	0.3034	N/A
120	sail/Sailor-4B	ViMedAQA_VI	Default_SP	0.1904	0.0540	0.3422	0.6257	0.3031	N/A
121	vilm/vinallama-2.7b-chat	ViMedAQA_VI	Empty_SP	0.1527	0.0518	0.3629	0.6292	0.2992	N/A
122	vilm/vinallama-2.7b-chat	Extract_VI	Empty_SP	0.1569	0.0527	0.3594	0.6144	0.2958	N/A
123	sail/Sailor-4B	Concise_VI	Default_SP	0.1575	0.0525	0.3529	0.6170	0.2950	N/A
124	sail/Sailor-4B	ViMedAQA_VI	Empty_SP	0.1556	0.0482	0.3453	0.6289	0.2945	N/A
"""

df = pd.read_csv(io.StringIO(data), sep='\t')

script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
script_name_base = os.path.splitext(os.path.basename(script_path))[0]
results_dir = os.path.join(script_dir, script_name_base)
os.makedirs(results_dir, exist_ok=True)

# Calculate the average 'Avg-Score' for each 'System_Prompt'
prompt_performance = df.groupby('System_Prompt')['Avg-Score'].mean().sort_values(ascending=False)

# Create the plot
plt.figure(figsize=(12, 8))
prompt_performance.plot(kind='bar')
plt.title('Performance Order of System Prompts')
plt.xlabel('System Prompt')
plt.ylabel('Average Avg-Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'SP_order.png'))
plt.close()


# Calculate the average 'Avg-Score' for each 'Model'
model_performance = df.groupby('Model')['Avg-Score'].mean().sort_values(ascending=False)

# Create the plot for model performance
plt.figure(figsize=(12, 8))
model_performance.plot(kind='bar')
plt.title('Performance of Each Model')
plt.xlabel('Model')
plt.ylabel('Average Avg-Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'Model_performance.png'))
plt.close()


# Get a list of unique models
unique_models = df['Model'].unique()

# Loop through each model to create a separate graph
for model in unique_models:
    # Filter the DataFrame for the current model and explicitly create a copy
    model_df = df[df['Model'] == model].copy()
    
    # Combine 'Prompt_Strategy' and 'System_Prompt' for unique x-axis labels
    model_df['Configuration'] = model_df['Prompt_Strategy'] + ' - ' + model_df['System_Prompt']
    
    # Sort the data by 'Avg-Score' for a clear performance ranking
    model_performance = model_df.set_index('Configuration')['Avg-Score'].sort_values(ascending=False)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    model_performance.plot(kind='bar')
    
    # Sanitize the model name to use it in the filename
    safe_model_name = model.replace('/', '_')
    
    plt.title(f'Performance Analysis for {model}')
    plt.xlabel('Prompt Configuration (Strategy - System Prompt)')
    plt.ylabel('Average Avg-Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot to a unique file for each model
    plt.savefig(os.path.join(results_dir, f'{safe_model_name}_performance.png'))
    plt.close()
    
# --- Visualization for each unique System_Prompt and Prompt_Strategy combination ---

# Group by both System_Prompt and Prompt_Strategy and calculate the mean Avg-Score
ps_performance = df.groupby(['System_Prompt', 'Prompt_Strategy'])['Avg-Score'].mean().reset_index()

# Create a new column for the combined label for better visualization
ps_performance['Combination'] = ps_performance['System_Prompt'] + ' - ' + ps_performance['Prompt_Strategy']

# Sort the combinations by performance in descending order
ps_performance = ps_performance.sort_values(by='Avg-Score', ascending=False)

# Create the plot
plt.figure(figsize=(20, 12)) # Increased figure size for better readability
ps_performance.plot(kind='bar', x='Combination', y='Avg-Score', legend=False)

plt.title('Performance of System Prompt and Prompt Strategy Combinations')
plt.xlabel('System Prompt - Prompt Strategy Combination')
plt.ylabel('Average Avg-Score')
plt.xticks(rotation=90, ha='center') # Rotate labels to be vertical
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(results_dir, 'SP_PS_combination_performance.png'))
plt.close()


# --- Visualization for the performance of each Prompt_Strategy ---

# Group by 'Prompt_Strategy' and calculate the mean 'Avg-Score'
strategy_performance = df.groupby('Prompt_Strategy')['Avg-Score'].mean().sort_values(ascending=False)

# Create the plot
plt.figure(figsize=(12, 8))
strategy_performance.plot(kind='bar')

plt.title('Performance of Each Prompt Strategy')
plt.xlabel('Prompt Strategy')
plt.ylabel('Average Avg-Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(results_dir, 'Prompt_Strategy_performance.png'))
plt.close()