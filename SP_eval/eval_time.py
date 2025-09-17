import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import sys
import re

# --- Data Loading and Parsing ---

# Paste the log data here
log_data = """
151.4s	44	==================================================
151.4s	45	Loading model: vilm/vietcuna-3b-v2
151.4s	46	==================================================
195.0s	47	
195.0s	48	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Default_SP] =====
215.5s	49	Time for generating answer: 20.42 seconds.
215.5s	50	
215.5s	51	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Default_SP] =====
236.6s	52	Time for generating answer: 21.14 seconds.
236.6s	53	
236.6s	54	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Default_SP] =====
246.2s	55	Time for generating answer: 9.60 seconds.
246.2s	56	
246.2s	57	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Default_SP] =====
289.2s	58	Time for generating answer: 43.03 seconds.
289.2s	59	
289.2s	60	===== Running Experiment: [Task: Full_VI] | [System Prompt: Default_SP] =====
299.4s	61	Time for generating answer: 10.23 seconds.
299.4s	62	
299.4s	63	===== Running Experiment: [Task: List_VI] | [System Prompt: Default_SP] =====
309.3s	64	Time for generating answer: 9.87 seconds.
309.3s	65	
309.3s	66	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Default_SP] =====
325.4s	67	Time for generating answer: 16.07 seconds.
325.4s	68	
325.4s	69	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Default_SP] =====
344.3s	70	Time for generating answer: 18.88 seconds.
344.3s	71	
344.3s	72	===== Running Experiment: [Task: Self-Contained] | [System Prompt: ViMedAQA_SP] =====
352.7s	73	Time for generating answer: 8.43 seconds.
352.7s	74	
352.7s	75	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Expert_SP] =====
372.7s	76	Time for generating answer: 20.01 seconds.
372.7s	77	
372.7s	78	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Expert_SP] =====
392.9s	79	Time for generating answer: 20.18 seconds.
392.9s	80	
392.9s	81	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Expert_SP] =====
403.3s	82	Time for generating answer: 10.37 seconds.
403.3s	83	
403.3s	84	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Expert_SP] =====
447.1s	85	Time for generating answer: 43.75 seconds.
447.1s	86	
447.1s	87	===== Running Experiment: [Task: Full_VI] | [System Prompt: Expert_SP] =====
456.5s	88	Time for generating answer: 9.48 seconds.
456.5s	89	
456.5s	90	===== Running Experiment: [Task: List_VI] | [System Prompt: Expert_SP] =====
464.8s	91	Time for generating answer: 8.34 seconds.
464.9s	92	
464.9s	93	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Expert_SP] =====
481.6s	94	Time for generating answer: 16.75 seconds.
481.6s	95	
481.6s	96	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Expert_SP] =====
501.9s	97	Time for generating answer: 20.33 seconds.
501.9s	98	
501.9s	99	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Empty_SP] =====
521.6s	100	Time for generating answer: 19.62 seconds.
521.6s	101	
521.6s	102	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Empty_SP] =====
541.4s	103	Time for generating answer: 19.86 seconds.
541.4s	104	
541.4s	105	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Empty_SP] =====
551.4s	106	Time for generating answer: 9.98 seconds.
551.4s	107	
551.4s	108	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Empty_SP] =====
599.0s	109	Time for generating answer: 47.58 seconds.
599.0s	110	
599.0s	111	===== Running Experiment: [Task: Full_VI] | [System Prompt: Empty_SP] =====
607.8s	112	Time for generating answer: 8.79 seconds.
607.8s	113	
607.8s	114	===== Running Experiment: [Task: List_VI] | [System Prompt: Empty_SP] =====
617.9s	115	Time for generating answer: 10.12 seconds.
617.9s	116	
617.9s	117	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Empty_SP] =====
632.4s	118	Time for generating answer: 14.43 seconds.
632.4s	119	
632.4s	120	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Empty_SP] =====
652.5s	121	Time for generating answer: 19.83 seconds.
652.7s	122	
652.7s	123	==================================================
652.7s	124	Loading model: arcee-ai/Arcee-VyLinh
652.7s	125	==================================================
795.6s	126	
795.6s	127	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Default_SP] =====
831.5s	128	Time for generating answer: 35.80 seconds.
831.5s	129	
831.5s	130	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Default_SP] =====
848.5s	131	Time for generating answer: 17.01 seconds.
848.5s	132	
848.5s	133	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Default_SP] =====
865.7s	134	Time for generating answer: 17.20 seconds.
865.7s	135	
865.7s	136	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Default_SP] =====
907.9s	137	Time for generating answer: 42.19 seconds.
907.9s	138	
907.9s	139	===== Running Experiment: [Task: Full_VI] | [System Prompt: Default_SP] =====
977.3s	140	Time for generating answer: 69.37 seconds.
977.3s	141	
977.3s	142	===== Running Experiment: [Task: List_VI] | [System Prompt: Default_SP] =====
1000.0s	143	Time for generating answer: 22.72 seconds.
1000.0s	144	
1000.0s	145	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Default_SP] =====
1017.6s	146	Time for generating answer: 17.63 seconds.
1017.6s	147	
1017.6s	148	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Default_SP] =====
1060.1s	149	Time for generating answer: 42.46 seconds.
1060.1s	150	
1060.1s	151	===== Running Experiment: [Task: Self-Contained] | [System Prompt: ViMedAQA_SP] =====
1109.0s	152	Time for generating answer: 48.95 seconds.
1109.0s	153	
1109.0s	154	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Expert_SP] =====
1152.4s	155	Time for generating answer: 43.39 seconds.
1152.4s	156	
1152.4s	157	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Expert_SP] =====
1169.5s	158	Time for generating answer: 17.09 seconds.
1169.5s	159	
1169.5s	160	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Expert_SP] =====
1186.7s	161	Time for generating answer: 17.15 seconds.
1186.7s	162	
1186.7s	163	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Expert_SP] =====
1223.1s	164	Time for generating answer: 36.46 seconds.
1223.1s	165	
1223.1s	166	===== Running Experiment: [Task: Full_VI] | [System Prompt: Expert_SP] =====
1309.5s	167	Time for generating answer: 86.37 seconds.
1309.5s	168	
1309.5s	169	===== Running Experiment: [Task: List_VI] | [System Prompt: Expert_SP] =====
1328.0s	170	Time for generating answer: 18.50 seconds.
1328.0s	171	
1328.0s	172	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Expert_SP] =====
1345.4s	173	Time for generating answer: 17.37 seconds.
1345.4s	174	
1345.4s	175	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Expert_SP] =====
1394.2s	176	Time for generating answer: 48.86 seconds.
1394.2s	177	
1394.2s	178	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Empty_SP] =====
1452.1s	179	Time for generating answer: 57.88 seconds.
1452.1s	180	
1452.1s	181	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Empty_SP] =====
1468.2s	182	Time for generating answer: 16.09 seconds.
1468.2s	183	
1468.2s	184	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Empty_SP] =====
1487.3s	185	Time for generating answer: 19.14 seconds.
1487.3s	186	
1487.3s	187	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Empty_SP] =====
1554.4s	188	Time for generating answer: 67.01 seconds.
1554.4s	189	
1554.4s	190	===== Running Experiment: [Task: Full_VI] | [System Prompt: Empty_SP] =====
1630.4s	191	Time for generating answer: 76.08 seconds.
1630.4s	192	
1630.4s	193	===== Running Experiment: [Task: List_VI] | [System Prompt: Empty_SP] =====
1684.8s	194	Time for generating answer: 54.37 seconds.
1684.8s	195	
1684.8s	196	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Empty_SP] =====
1700.2s	197	Time for generating answer: 15.44 seconds.
1700.2s	198	
1700.2s	199	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Empty_SP] =====
1765.9s	200	Time for generating answer: 65.44 seconds.
1766.1s	201	
1766.1s	202	==================================================
1766.1s	203	Loading model: alpha-ai/LLAMA3-3B-Medical-COT
1766.1s	204	==================================================
1780.8s	205	
1780.8s	206	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Default_SP] =====
1793.4s	207	Time for generating answer: 12.55 seconds.
1793.4s	208	
1793.4s	209	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Default_SP] =====
1798.4s	210	Time for generating answer: 4.96 seconds.
1798.4s	211	
1798.4s	212	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Default_SP] =====
1803.4s	213	Time for generating answer: 5.02 seconds.
1803.4s	214	
1803.4s	215	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Default_SP] =====
1814.0s	216	Time for generating answer: 10.63 seconds.
1814.0s	217	
1814.0s	218	===== Running Experiment: [Task: Full_VI] | [System Prompt: Default_SP] =====
1831.7s	219	Time for generating answer: 17.65 seconds.
1831.7s	220	
1831.7s	221	===== Running Experiment: [Task: List_VI] | [System Prompt: Default_SP] =====
1841.5s	222	Time for generating answer: 9.87 seconds.
1841.5s	223	
1841.5s	224	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Default_SP] =====
1848.6s	225	Time for generating answer: 7.09 seconds.
1848.6s	226	
1848.6s	227	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Default_SP] =====
1869.0s	228	Time for generating answer: 20.34 seconds.
1869.0s	229	
1869.0s	230	===== Running Experiment: [Task: Self-Contained] | [System Prompt: ViMedAQA_SP] =====
1878.4s	231	Time for generating answer: 9.46 seconds.
1878.4s	232	
1878.4s	233	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Expert_SP] =====
1891.8s	234	Time for generating answer: 13.34 seconds.
1891.8s	235	
1891.8s	236	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Expert_SP] =====
1896.5s	237	Time for generating answer: 4.74 seconds.
1896.5s	238	
1896.5s	239	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Expert_SP] =====
1901.9s	240	Time for generating answer: 5.38 seconds.
1901.9s	241	
1901.9s	242	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Expert_SP] =====
1913.8s	243	Time for generating answer: 11.90 seconds.
1913.8s	244	
1913.8s	245	===== Running Experiment: [Task: Full_VI] | [System Prompt: Expert_SP] =====
1936.8s	246	Time for generating answer: 22.93 seconds.
1936.8s	247	
1936.8s	248	===== Running Experiment: [Task: List_VI] | [System Prompt: Expert_SP] =====
1953.2s	249	Time for generating answer: 16.41 seconds.
1953.2s	250	
1953.2s	251	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Expert_SP] =====
1961.2s	252	Time for generating answer: 8.07 seconds.
1961.2s	253	
1961.2s	254	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Expert_SP] =====
1975.0s	255	Time for generating answer: 13.75 seconds.
1975.0s	256	
1975.0s	257	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Empty_SP] =====
1981.5s	258	Time for generating answer: 6.48 seconds.
1981.5s	259	
1981.5s	260	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Empty_SP] =====
1986.3s	261	Time for generating answer: 4.86 seconds.
1986.3s	262	
1986.3s	263	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Empty_SP] =====
1990.7s	264	Time for generating answer: 4.38 seconds.
1990.7s	265	
1990.7s	266	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Empty_SP] =====
1999.8s	267	Time for generating answer: 9.13 seconds.
1999.8s	268	
1999.8s	269	===== Running Experiment: [Task: Full_VI] | [System Prompt: Empty_SP] =====
2011.0s	270	Time for generating answer: 11.13 seconds.
2011.0s	271	
2011.0s	272	===== Running Experiment: [Task: List_VI] | [System Prompt: Empty_SP] =====
2022.1s	273	Time for generating answer: 11.16 seconds.
2022.1s	274	
2022.1s	275	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Empty_SP] =====
2027.4s	276	Time for generating answer: 5.27 seconds.
2027.4s	277	
2027.4s	278	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Empty_SP] =====
2044.2s	279	Time for generating answer: 16.57 seconds.
2044.4s	280	
2044.4s	281	==================================================
2044.4s	282	Loading model: vilm/vinallama-2.7b-chat
2044.4s	283	==================================================
2081.2s	284	
2081.2s	285	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Default_SP] =====
2174.0s	286	Time for generating answer: 92.76 seconds.
2174.0s	287	
2174.0s	288	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Default_SP] =====
2258.0s	289	Time for generating answer: 83.96 seconds.
2258.0s	290	
2258.0s	291	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Default_SP] =====
2341.2s	292	Time for generating answer: 83.16 seconds.
2341.2s	293	
2341.2s	294	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Default_SP] =====
2428.9s	295	Time for generating answer: 87.77 seconds.
2428.9s	296	
2428.9s	297	===== Running Experiment: [Task: Full_VI] | [System Prompt: Default_SP] =====
2511.9s	298	Time for generating answer: 82.94 seconds.
2511.9s	299	
2511.9s	300	===== Running Experiment: [Task: List_VI] | [System Prompt: Default_SP] =====
2594.0s	301	Time for generating answer: 82.12 seconds.
2594.0s	302	
2594.0s	303	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Default_SP] =====
2676.7s	304	Time for generating answer: 82.71 seconds.
2676.7s	305	
2676.7s	306	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Default_SP] =====
2757.6s	307	Time for generating answer: 80.92 seconds.
2757.6s	308	
2757.6s	309	===== Running Experiment: [Task: Self-Contained] | [System Prompt: ViMedAQA_SP] =====
2838.3s	310	Time for generating answer: 80.68 seconds.
2838.3s	311	
2838.3s	312	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Expert_SP] =====
2919.4s	313	Time for generating answer: 81.05 seconds.
2919.4s	314	
2919.4s	315	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Expert_SP] =====
3001.1s	316	Time for generating answer: 81.70 seconds.
3001.1s	317	
3001.1s	318	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Expert_SP] =====
3083.1s	319	Time for generating answer: 81.98 seconds.
3083.1s	320	
3083.1s	321	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Expert_SP] =====
3169.8s	322	Time for generating answer: 86.79 seconds.
3169.8s	323	
3169.8s	324	===== Running Experiment: [Task: Full_VI] | [System Prompt: Expert_SP] =====
3251.8s	325	Time for generating answer: 82.00 seconds.
3251.8s	326	
3251.8s	327	===== Running Experiment: [Task: List_VI] | [System Prompt: Expert_SP] =====
3333.3s	328	Time for generating answer: 81.42 seconds.
3333.3s	329	
3333.3s	330	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Expert_SP] =====
3415.7s	331	Time for generating answer: 82.48 seconds.
3415.7s	332	
3415.7s	333	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Expert_SP] =====
3497.5s	334	Time for generating answer: 81.70 seconds.
3497.5s	335	
3497.5s	336	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Empty_SP] =====
3578.1s	337	Time for generating answer: 80.63 seconds.
3578.1s	338	
3578.1s	339	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Empty_SP] =====
3658.9s	340	Time for generating answer: 80.83 seconds.
3658.9s	341	
3658.9s	342	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Empty_SP] =====
3742.3s	343	Time for generating answer: 83.40 seconds.
3742.3s	344	
3742.3s	345	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Empty_SP] =====
3831.4s	346	Time for generating answer: 89.10 seconds.
3831.4s	347	
3831.4s	348	===== Running Experiment: [Task: Full_VI] | [System Prompt: Empty_SP] =====
3914.8s	349	Time for generating answer: 83.38 seconds.
3914.8s	350	
3914.8s	351	===== Running Experiment: [Task: List_VI] | [System Prompt: Empty_SP] =====
3998.0s	352	Time for generating answer: 83.09 seconds.
3998.0s	353	
3998.0s	354	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Empty_SP] =====
4081.0s	355	Time for generating answer: 83.13 seconds.
4081.0s	356	
4081.0s	357	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Empty_SP] =====
4163.6s	358	Time for generating answer: 82.36 seconds.
4163.8s	359	
4163.8s	360	==================================================
4163.8s	361	Loading model: sail/Sailor-4B
4163.8s	362	==================================================
4240.9s	363	
4240.9s	364	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Default_SP] =====
4351.3s	365	Time for generating answer: 110.37 seconds.
4351.3s	366	
4351.3s	367	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Default_SP] =====
4449.1s	368	Time for generating answer: 97.83 seconds.
4449.1s	369	
4449.1s	370	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Default_SP] =====
4558.9s	371	Time for generating answer: 109.72 seconds.
4558.9s	372	
4558.9s	373	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Default_SP] =====
4670.8s	374	Time for generating answer: 111.90 seconds.
4670.8s	375	
4670.8s	376	===== Running Experiment: [Task: Full_VI] | [System Prompt: Default_SP] =====
4779.5s	377	Time for generating answer: 108.72 seconds.
4779.5s	378	
4779.5s	379	===== Running Experiment: [Task: List_VI] | [System Prompt: Default_SP] =====
4883.2s	380	Time for generating answer: 103.73 seconds.
4883.2s	381	
4883.2s	382	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Default_SP] =====
4977.6s	383	Time for generating answer: 94.33 seconds.
4977.6s	384	
4977.6s	385	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Default_SP] =====
5069.1s	386	Time for generating answer: 91.57 seconds.
5069.1s	387	
5069.1s	388	===== Running Experiment: [Task: Self-Contained] | [System Prompt: ViMedAQA_SP] =====
5155.9s	389	Time for generating answer: 86.80 seconds.
5155.9s	390	
5155.9s	391	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Expert_SP] =====
5240.3s	392	Time for generating answer: 84.35 seconds.
5240.3s	393	
5240.3s	394	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Expert_SP] =====
5356.4s	395	Time for generating answer: 116.15 seconds.
5356.4s	396	
5356.4s	397	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Expert_SP] =====
5443.8s	398	Time for generating answer: 87.41 seconds.
5443.8s	399	
5443.8s	400	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Expert_SP] =====
5563.2s	401	Time for generating answer: 119.32 seconds.
5563.2s	402	
5563.2s	403	===== Running Experiment: [Task: Full_VI] | [System Prompt: Expert_SP] =====
5670.9s	404	Time for generating answer: 107.69 seconds.
5670.9s	405	
5670.9s	406	===== Running Experiment: [Task: List_VI] | [System Prompt: Expert_SP] =====
5759.9s	407	Time for generating answer: 89.03 seconds.
5759.9s	408	
5759.9s	409	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Expert_SP] =====
5860.0s	410	Time for generating answer: 100.09 seconds.
5860.0s	411	
5860.0s	412	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Expert_SP] =====
5946.1s	413	Time for generating answer: 86.07 seconds.
5946.1s	414	
5946.1s	415	===== Running Experiment: [Task: Direct_VI] | [System Prompt: Empty_SP] =====
6041.6s	416	Time for generating answer: 95.53 seconds.
6041.6s	417	
6041.6s	418	===== Running Experiment: [Task: Extract_VI] | [System Prompt: Empty_SP] =====
6149.6s	419	Time for generating answer: 108.03 seconds.
6149.6s	420	
6149.6s	421	===== Running Experiment: [Task: Concise_VI] | [System Prompt: Empty_SP] =====
6252.8s	422	Time for generating answer: 103.17 seconds.
6252.8s	423	
6252.8s	424	===== Running Experiment: [Task: Few_Shot_VI] | [System Prompt: Empty_SP] =====
6365.9s	425	Time for generating answer: 113.09 seconds.
6365.9s	426	
6365.9s	427	===== Running Experiment: [Task: Full_VI] | [System Prompt: Empty_SP] =====
6477.0s	428	Time for generating answer: 111.09 seconds.
6477.0s	429	
6477.0s	430	===== Running Experiment: [Task: List_VI] | [System Prompt: Empty_SP] =====
6579.5s	431	Time for generating answer: 102.50 seconds.
6579.5s	432	
6579.5s	433	===== Running Experiment: [Task: Strict_Rules_VI] | [System Prompt: Empty_SP] =====
6661.5s	434	Time for generating answer: 82.01 seconds.
6661.5s	435	
6661.5s	436	===== Running Experiment: [Task: ViMedAQA_VI] | [System Prompt: Empty_SP] =====
6771.9s	437	Time for generating answer: 110.17 seconds.
"""

# Define the directory to save plots
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
script_name_base = os.path.splitext(os.path.basename(script_path))[0]
results_dir = os.path.join(script_dir, script_name_base)
os.makedirs(results_dir, exist_ok=True)

# Regex patterns to extract information
model_pattern = re.compile(r"Loading model: (.*)")
exp_pattern = re.compile(r"\[Task: (.*?)\] \| \[System Prompt: (.*?)\]")
time_pattern = re.compile(r"Time for generating answer: ([\d.]+) seconds.")

# Parse the log data
parsed_data = []
current_model = None
current_exp = None

for line in log_data.strip().split('\n'):
    model_match = model_pattern.search(line)
    if model_match:
        current_model = model_match.group(1).strip()
        continue

    exp_match = exp_pattern.search(line)
    if exp_match:
        current_exp = {
            "Model": current_model,
            "Prompt_Strategy": exp_match.group(1).strip(),
            "System_Prompt": exp_match.group(2).strip()
        }
        continue

    time_match = time_pattern.search(line)
    if time_match and current_exp:
        current_exp["Inference_Time"] = float(time_match.group(1))
        parsed_data.append(current_exp)
        current_exp = None

# Create a DataFrame
time_df = pd.DataFrame(parsed_data)


# --- Graph 1: Inference Time by Prompt Strategy ---

# Group by 'Prompt_Strategy' and calculate the mean 'Inference_Time'
strategy_time_performance = time_df.groupby('Prompt_Strategy')['Inference_Time'].mean().sort_values(ascending=False)

# Create the plot
plt.figure(figsize=(12, 8))
strategy_time_performance.plot(kind='bar', color='skyblue')
plt.title('Average Inference Time by Prompt Strategy')
plt.xlabel('Prompt Strategy')
plt.ylabel('Average Inference Time (seconds)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(results_dir, 'Prompt_Strategy_inference_time.png'))
plt.close()

print(f"Saved Prompt Strategy inference time graph to: {os.path.join(results_dir, 'Prompt_Strategy_inference_time.png')}")


# --- Graph 2: Inference Time by System Prompt ---

# Group by 'System_Prompt' and calculate the mean 'Inference_Time'
system_prompt_time_performance = time_df.groupby('System_Prompt')['Inference_Time'].mean().sort_values(ascending=False)

# Create the plot
plt.figure(figsize=(10, 7))
system_prompt_time_performance.plot(kind='bar', color='lightcoral')
plt.title('Average Inference Time by System Prompt')
plt.xlabel('System Prompt')
plt.ylabel('Average Inference Time (seconds)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(results_dir, 'System_Prompt_inference_time.png'))
plt.close()

print(f"Saved System Prompt inference time graph to: {os.path.join(results_dir, 'System_Prompt_inference_time.png')}")


# --- Graph 3: Inference Time by Model ---

# Group by 'Model' and calculate the mean 'Inference_Time'
model_time_performance = time_df.groupby('Model')['Inference_Time'].mean().sort_values(ascending=False)

# Create the plot
plt.figure(figsize=(14, 9)) # Increased figure size for better model name visibility
model_time_performance.plot(kind='bar', color='lightgreen')
plt.title('Average Inference Time by Model')
plt.xlabel('Model')
plt.ylabel('Average Inference Time (seconds)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(results_dir, 'Model_inference_time.png'))
plt.close()

print(f"Saved Model inference time graph to: {os.path.join(results_dir, 'Model_inference_time.png')}")