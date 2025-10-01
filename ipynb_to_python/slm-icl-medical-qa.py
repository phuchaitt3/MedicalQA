#!/usr/bin/env python
# coding: utf-8

# In[1]:


FIRST_RUN_MODEL = True


# In[2]:


# Step 1: Install necessary libraries
get_ipython().system('pip install -q transformers datasets accelerate bitsandbytes torch evaluate rouge_score sentencepiece bert_score')


# In[3]:


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from datasets import load_dataset
import pandas as pd
import random
import evaluate
import warnings
import time

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()


# In[4]:


# Step 2: Authenticate with Hugging Face
from kaggle_secrets import UserSecretsClient
try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HUGGING_FACE_TOKEN")
except Exception as e:
    print("Could not retrieve Hugging Face token. Please ensure it is stored as a Kaggle secret named 'HUGGING_FACE_TOKEN'.")
    hf_token = None


# In[5]:


# --- UPDATED: Step 1 - Define Bilingual Prompt Engineering Strategies ---
generation_times = {}

def create_prompt_and_get_config(model_id, tokenizer, system_prompt, full_instruction_text):
    """
    A single, unified function to create a model-specific prompt and return
    the associated answer start tag for parsing.

    Returns:
        tuple: (formatted_prompt_string, answer_start_tag_string)
    """  
    # Models WITHOUT a dedicated System Prompt
    # For these, we combine the system and user prompts into a single instruction.
    if model_id in ["vilm/vietcuna-3b-v2", "sail/Sailor-4B"]:
        # If a system prompt is provided, prepend it.
        if system_prompt:
            combined_instruction = f"{system_prompt}\n\n{full_instruction_text}"
        # Otherwise, just use the user instruction directly.
        else:
            combined_instruction = full_instruction_text

        if model_id == "vilm/vietcuna-3b-v2":
            prompt = f"A chat between a curious user and an artificial intelligence assistant.\nUSER: {combined_instruction}\nASSISTANT:"
            answer_start_tag = "ASSISTANT:"
            return prompt, answer_start_tag

        # https://huggingface.co/sail/Sailor-4B
        if model_id == "sail/Sailor-4B":
            prompt = f"{combined_instruction}\n\nCâu trả lời:"
            answer_start_tag = ""
            return prompt, answer_start_tag

    # For chat models, we build the message list programmatically.
    messages = []
    # Only add the system role if the system_prompt is not empty.
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add the user/question role
    if model_id == "sail/Sailor-4B-Chat":
        messages.append({"role": "question", "content": full_instruction_text})
    else:
        # Default to "user" role for all other chat models
        messages.append({"role": "user", "content": full_instruction_text})
    
    if model_id == "arcee-ai/Arcee-VyLinh":
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        answer_start_tag = "<|im_start|>assistant"
        return prompt, answer_start_tag

    if model_id == "alpha-ai/LLAMA3-3B-Medical-COT":
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        answer_start_tag = "<|start_header_id|>assistant<|end_header_id|>"
        return prompt, answer_start_tag
    
    # https://huggingface.co/sail/Sailor-4B-Chat
    if model_id == "sail/Sailor-4B-Chat":
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        answer_start_tag = "answer:"
        return prompt, answer_start_tag

    if "vilm/vinallama-2.7b" in model_id:
        # This model's template is custom and doesn't use apply_chat_template,
        # so we handle it separately.
        if system_prompt:
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{full_instruction_text}<|im_end|>\n"
                f"<|im_start|>assistant"
            )
        else:
            # Version without a system prompt
            prompt = (
                f"<|im_start|>user\n{full_instruction_text}<|im_end|>\n"
                f"<|im_start|>assistant"
            )
        answer_start_tag = "<|im_start|>assistant"
        return prompt, answer_start_tag

    # Nothing matches
    return "", ""


# In[6]:


def clean_answer_tag(raw_answer, answer_tag):
    if answer_tag == "":
        if raw_answer != "SKIPPED":
            print("Info: empty answer_tag")
            print(f"raw_answer: {raw_answer}\n")
        return raw_answer.strip()
        
    clean_answer = raw_answer.split(answer_tag)[-1].strip() if answer_tag and answer_tag in raw_answer else raw_answer.strip()
    return clean_answer


# In[7]:


LOAD_FROM_VIMEDAQA = True

# --- Common settings ---
dataset_id = "tmnam20/ViMedAQA"
comparison_csv_path = "/kaggle/input/complex-truth/complex_samples_with_ground_truth.csv"

model_ids = [
    "arcee-ai/Arcee-VyLinh",
]
RUN_MODEL = True
PRINT_PROMPT = False
PROVIDE_CONTEXT = True


# In[14]:


if LOAD_FROM_VIMEDAQA:
    EVAL_FULL_DATASET = True
    
    # seed_num = 1
    # NUM_SAMPLES_INITIAL = 20
    
    seed_num = 6
    NUM_SAMPLES_INITIAL = 50
    
    # seed_num = 3
    # NUM_SAMPLES_INITIAL = 20
    
    ENABLE_SUBSET_SAMPLING = False
    NUM_SAMPLES_FINAL = 50
    
    ENABLE_SINGLE_INDEX_SELECTION = False
    TARGET_INDEX = 0 # The index (starting from 0) of the sample to select.
    
    # Step 4: Load and Prepare the Dataset
    try:
        dataset = load_dataset(dataset_id, split="test")
        print(f"Dataset loaded successfully! Total samples: {len(dataset)}")
    
        if EVAL_FULL_DATASET:
            eval_dataset = dataset
        else:
            # --- First Sampling Step: Always get the initial samples ---
            random.seed(seed_num) # for reproducibility
            initial_random_indices = random.sample(range(len(dataset)), NUM_SAMPLES_INITIAL)
            initial_eval_dataset = dataset.select(initial_random_indices)
        
            print(f"Created an initial random evaluation set with {len(initial_eval_dataset)} samples.")
        
            # --- Conditional second sampling/selection ---
            if ENABLE_SUBSET_SAMPLING:
                print("Subset sampling is ENABLED. Performing second randomization...")
                # Re-seed to ensure this step is also reproducible
                random.seed(seed_num)
                final_random_indices = random.sample(range(len(initial_eval_dataset)), NUM_SAMPLES_FINAL)
                # Final dataset is the smaller, 50-sample subset
                eval_dataset = initial_eval_dataset.select(final_random_indices)
                print(f"Further randomized and reduced the set to a final size of {len(eval_dataset)} samples.")
                
            elif ENABLE_SINGLE_INDEX_SELECTION:
                print(f"Single index selection is ENABLED. Subsetting to 1 sample at index: {TARGET_INDEX}...")
                # Ensure the target index is valid
                if 0 <= TARGET_INDEX < len(initial_eval_dataset):
                    # Final dataset is the single selected sample
                    eval_dataset = initial_eval_dataset.select([TARGET_INDEX])
                    print(f"Successfully created a final dataset with {len(eval_dataset)} sample.")
                else:
                    raise IndexError(f"TARGET_INDEX {TARGET_INDEX} is out of bounds for the initial sample size of {len(initial_eval_dataset)}.")
    
            else:
                # Final dataset is the larger, initial sample set
                eval_dataset = initial_eval_dataset
    
    except Exception as e:
        print(f"Failed to load or process the dataset. Error: {e}")
        eval_dataset = None


# In[15]:


if not LOAD_FROM_VIMEDAQA:
    seed_num = 6
    NUM_SAMPLES_INITIAL = 5 # or 44
    RUN_FULL = True
    
    from datasets import Dataset
    try:
        # Load the CSV into a pandas DataFrame
        comparison_df = pd.read_csv("/kaggle/input/complex-truth/complex_samples_with_ground_truth.csv")
    
        if 'Ground_Truth_Answer' in comparison_df.columns:
            comparison_df.rename(columns={'Ground_Truth_Answer': 'answer'}, inplace=True)
            
        if 'Question' in comparison_df.columns:
            comparison_df.rename(columns={'Question': 'question'}, inplace=True)
    
        if 'Context' in comparison_df.columns:
            comparison_df.rename(columns={'Context': 'context'}, inplace=True)
    
        # Convert the pandas DataFrame to a Hugging Face Dataset object for compatibility
        eval_dataset = Dataset.from_pandas(comparison_df)
    
        if not RUN_FULL:
            random.seed(seed_num) # for reproducibility
            initial_random_indices = random.sample(range(len(eval_dataset)), NUM_SAMPLES_INITIAL)
            initial_eval_dataset = eval_dataset.select(initial_random_indices)
            eval_dataset = initial_eval_dataset
            
        print(eval_dataset)
    
    except FileNotFoundError:
        print(f"Error: The file '{comparison_csv_path}' was not found. Please ensure it is in the correct directory.")
        eval_dataset = None
    except Exception as e:
        print(f"An error occurred: {e}")
        eval_dataset = None


# In[16]:


USER_PROMPTS = {
    "Concise_EN": (
        f"Based ONLY on the text in the Context below, answer the Question. "
        f"Your answer must be concise, to the point, and contain no information not present in the text. "
        f"Do NOT add any explanation."
    ),
}

SYSTEM_PROMPTS = [
    (
        "Expert_SP_EN",
        "You are a medical expert AI. Based on your expertise, answer the following Question in Vietnamese, using ONLY the provided Context.",
        "{base_instruction}\n\n### Context:\n{context}\n\n### Question:\n{question}",
    ),
]

def generate_direct_answer(sample, model_id, tokenizer, text_generator):
    """
    Generates an answer directly without any guiding questions (The SIMPLE path).
    """
    base_instruction = USER_PROMPTS.get("Concise_EN")
    SP_name, direct_system_prompt, instruction_format = SYSTEM_PROMPTS[0]
    full_instruction_text = instruction_format.format(
                            base_instruction=base_instruction, 
                            context=sample['context'], 
                            question=sample['question']
                        )
    
    direct_prompt, tag = create_prompt_and_get_config(model_id, tokenizer, direct_system_prompt, full_instruction_text)
    if PRINT_PROMPT:
        print(f"direct_prompt: {direct_prompt}\n\n")
    
    raw_answer = text_generator(direct_prompt, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
    reasoning_log = "Strategy: SIMPLE. No guiding questions were generated."
    
    return raw_answer, tag, reasoning_log


# In[17]:


# PRINT_PROMPT = False
# PRINT_PROMPT = PRINT_PROMT

REASONER_SYSTEM_PROMPT = "You are a medical AI expert that excels at breaking down a complex medical Context and extracting an answer from the Context to answer a medical Question."
STOP_AT_QA = False
PRINT_QA = False
CHECK_GUIDING = False
if PRINT_PROMPT:
    PRINT_QA = True
print(f"PRINT_QA: {PRINT_QA}\n")

def generate_with_internal_reasoning(sample, model_id, tokenizer, text_generator, system_prompt):
    """
    Performs a multi-step internal reasoning process to generate a final answer.
    """
    # --- Step 1: Generate Internal Queries ---
    # ask_prompt_text = (
    #     "Based ONLY on the text in the Context below, generate a concise, numbered list of all the specific sub-questions "
    #     "you must answer first in order to comprehensively answer the Main Question. "
    #     "Sub-questions should be formatted as standard questions, not instructions."
    #     "Do not answer them yet.\n\n"
    #     f"### Context:\n{sample['context']}\n\n### Main Question:\n{sample['question']}"
    # )

    ask_prompt_text = (
        "Based ONLY on the text in the Context below, generate a concise, numbered list of up to 3 Guiding Questions. "
        "These questions should act as a checklist to help you find all the relevant pieces of information needed to comprehensively answer the Main Question, especially if the answer is scattered across a long Context. "
        "If the answer is straightforward and located in a single, clear spot, respond with 'No guiding questions needed.', do not give Guiding Question list.\n\n"
        f"### Context:\n{sample['context']}\n\n### Main Question:\n{sample['question']}\n\n"
        "### Numbered list of Guiding Questions (or 'No guiding questions needed.'):\n"
    )
    
    if PRINT_QA or CHECK_GUIDING:
        print(f"- ask_prompt_text: {ask_prompt_text}\n")

    ask_prompt_formatted, _ = create_prompt_and_get_config(model_id, tokenizer, system_prompt, ask_prompt_text)
    
    generated_output = text_generator(ask_prompt_formatted, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
    if PRINT_QA:
        print(f"- generated_output: {generated_output}\n")
    raw_queries = generated_output.replace(ask_prompt_formatted, "").strip()

    if "no guiding questions" in raw_queries.lower():
        print("  > Model determined no sub-questions needed.")
        # return generate_direct_answer(sample, model_id, tokenizer, text_generator)
        raw_answer = "SKIPPED"
        tag = ""
        reasoning_log = ""
        
        return raw_answer, tag, reasoning_log
    
    internal_queries = [q.strip().lstrip('0123456789.- ').rstrip() for q in raw_queries.split('\n') if q.strip()]
    
    if not internal_queries:
        direct_answer_prompt, tag = create_prompt_and_get_config(model_id, tokenizer, system_prompt, f"Context: {sample['context']}\n\nQuestion: {sample['question']}\n\nAnswer:")
        final_answer_raw = text_generator(direct_answer_prompt, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
        return final_answer_raw, tag, "FALLBACK: No internal queries were generated."

    internal_qa_log = ""
    # --- Step 2: Answer Each Internal Query ---
    for query in internal_queries:
        if not query: continue
        # finder_prompt_text = f"Based ONLY on the Context below, answer the following Question concisely.\n\n### Context:\n{sample['context']}\n\n### Question:\n{query}"
        # finder_prompt_formatted, _ = create_prompt_and_get_config(model_id, tokenizer, system_prompt, finder_prompt_text)
        
        # internal_answer_raw = text_generator(finder_prompt_formatted, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
        # internal_answer = internal_answer_raw.replace(finder_prompt_formatted, "").strip()

        # Use the original context but the new guiding question.
        temp_sample = {
            'context': sample['context'],
            'question': query
        }

        internal_answer_raw, internal_answer_tag, _ = generate_direct_answer(
            sample=temp_sample, 
            model_id=model_id, 
            tokenizer=tokenizer, 
            text_generator=text_generator
        )

        internal_answer = clean_answer_tag(internal_answer_raw, internal_answer_tag)

        qa_pair = f"Guiding Question: {query}\nGuiding Answer: {internal_answer}\n\n"
        if PRINT_QA:
            print(f"- qa_pair: {qa_pair}")
        internal_qa_log += qa_pair

    if not STOP_AT_QA:
        # --- Step 3: Synthesize the Final Answer ---
        # synthesis_prompt_text = (
        #     f"You have been provided with a Context, a Main Question, and a series of Guiding Question & Answer pairs that might help answer the Main Question.\n"
        #     f"Your TASK is to generate a Final Answer to the Main Question, using the provided information.\n"
        #     f"Your Final Answer MUST use exact phrases from the Context. Guiding Q&A pairs only help to locate relevant information in the Context to answer the Main Question, "
        #     f"and should NOT be cited in the Final Answer. "
        #     f"Do not add any extra words or explanation not in the Context.\n"
        #     f"Begin the Final Answer by rephrasing the Main Question as a declarative statement.\n"
        #     f"Your Final Answer must use Vietnamese. Ensure that no English words are included.\n\n"
    
        #     f"### Context:\n{sample['context']}\n\n"
        #     f"### Main Question:\n{sample['question']}\n\n"
        #     f"### Guiding Q&A:\n{internal_qa_log.strip()}\n\n"
        #     f"### Final Answer:"
        # )

        synthesis_prompt_text = (
            "Use the Context and Guiding Q&A to answer the Main Question in Vietnamese.\n\n"
            "**Instructions:**\n"
            "- Use only information from the Context.\n"
            "- Do not reference the Guiding Q&A in the final answer.\n"
            "- Start the answer by rephrasing the Main Question.\n\n"
            f"### Context:\n{sample['context']}\n\n"
            f"### Main Question:\n{sample['question']}\n\n"
            f"### Guiding Q&A:\n{internal_qa_log.strip()}\n\n"
            "### Final Answer:"
        )
    
        synthesis_prompt_formatted, final_tag = create_prompt_and_get_config(model_id, tokenizer, system_prompt, synthesis_prompt_text)
        if PRINT_PROMPT:
            print(synthesis_prompt_formatted)
    
        final_answer_raw = text_generator(synthesis_prompt_formatted, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
        
        return final_answer_raw, final_tag, internal_qa_log.strip()
    else:
        return None, None, None


# In[18]:


SKIP_ENTIRE_SAMPLE_IF_DEEMED_SIMPLE = True

# Set to True to stop the generation process after enough complex samples are found.
ENABLE_MAX_COMPLEX_LIMIT = True
# Define the maximum number of complex samples you want to collect before stopping.
MAX_COMPLEX_SAMPLES_TO_COLLECT = 20


# In[19]:


# Step 5: Generate Answers from Each Model
all_generated_answers = {}
intermediate_results = [] # Use a list of dicts for easier DataFrame creation
complex_samples_to_save = []

if eval_dataset and hf_token:
    # Loop through each model to generate answers
    for model_id in model_ids:
        print("\n" + "="*50)
        print(f"Loading model: {model_id}")
        print("="*50)

        try:
            # prompt_name = f"Dynamic Follow-up"
            # sp_name = f"Dynamic Follow-up"
            # result_key_tuple = (model_id, prompt_name, sp_name)
            
            if RUN_MODEL:
                if FIRST_RUN_MODEL:
                    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=False,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        token=hf_token,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    text_generator = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
                    FIRST_RUN_MODEL = False

                evaluation_ground_truths = [] 
                simple_path_answers = []
                complex_path_answers = []
                simple_total_time = 0.0
                complex_total_time = 0.0
                
                model_answers = []
                reasoning_logs = []

                for i, sample in enumerate(eval_dataset):
                    print("\n" + '-'*10)
                    print(f"Processing sample {i+1}/{len(eval_dataset)}...")

                    # --- 1. Run COMPLEX Path ---
                    start_complex = time.time()
                    # Using the reasoning function that dynamically determines the number of questions
                    complex_raw, complex_tag, complex_log = generate_with_internal_reasoning(
                        sample, model_id, tokenizer, text_generator, REASONER_SYSTEM_PROMPT
                    )
                    end_complex = time.time()
                    
                    complex_clean = clean_answer_tag(complex_raw, complex_tag)

                    if complex_clean != "SKIPPED" and LOAD_FROM_VIMEDAQA:
                        complex_samples_to_save.append({
                            'context': sample['context'],
                            'question': sample['question'],
                            'Chosen_Strategy': 'COMPLEX',        # Set the strategy
                            'Model_Reason': complex_log,           # Use the generated reasoning log
                            'Ground_Truth_Answer': sample['answer'] # Use the correct final column name
                        })
                        print(f"  > Sample identified as COMPLEX. Total collected: {len(complex_samples_to_save)}/{MAX_COMPLEX_SAMPLES_TO_COLLECT}")

                        if ENABLE_MAX_COMPLEX_LIMIT and len(complex_samples_to_save) >= MAX_COMPLEX_SAMPLES_TO_COLLECT:
                            print(f"\nTarget of {MAX_COMPLEX_SAMPLES_TO_COLLECT} complex samples reached. Halting further processing.")
                            break # Exit the loop over the dataset
                    
                    if SKIP_ENTIRE_SAMPLE_IF_DEEMED_SIMPLE and complex_clean == "SKIPPED":
                        print(f"  > COMPLEX path was SKIPPED. Skipping this entire sample.")
                        continue
                    
                    # --- 2. Run SIMPLE Path ---
                    start_simple = time.time()
                    simple_raw, simple_tag, simple_log = generate_direct_answer(
                        sample, model_id, tokenizer, text_generator
                    )
                    end_simple = time.time()
                    
                    simple_clean = clean_answer_tag(simple_raw, simple_tag)

                    # --- 3. Accumulate timings and answers ONLY for non-skipped samples ---
                    simple_total_time += (end_simple - start_simple)
                    complex_total_time += (end_complex - start_complex) # Timings are still valid
                    
                    simple_path_answers.append(simple_clean)
                    complex_path_answers.append(complex_clean)

                    evaluation_ground_truths.append([sample['answer']]) # Keep the list-of-lists format
                    
                    # --- 4. Store side-by-side results for easy visual comparison ---
                    intermediate_results.append({
                        "Sample_ID": i,
                        "Model": model_id,
                        "Question": sample['question'],
                        "Context": sample['context'],
                        "Ground_Truth_Answer": sample['answer'],
                        "Answer_SIMPLE_Path": simple_clean,
                        "Answer_COMPLEX_Path": complex_clean,
                        "Reasoning_Log_COMPLEX": complex_log
                    })
                
                # Define descriptive system prompt names for each path
                simple_sp_name = "Concise_EN"
                complex_sp_name = "FollowUp_EN"
            
                # Create the 3-element keys
                simple_key = (model_id, "SIMPLE_PATH", simple_sp_name)
                complex_key = (model_id, "COMPLEX_PATH", complex_sp_name)
            
                # Populate the answers dictionary
                all_generated_answers[simple_key] = simple_path_answers
                all_generated_answers[complex_key] = complex_path_answers
            
                # Also populate the generation_times dictionary to prevent a future error in the eval cell
                # We'll assign the total time to both, as they were run in the same batch.
                generation_times[simple_key] = simple_total_time
                generation_times[complex_key] = complex_total_time
                
                # Save the detailed side-by-side comparison results
                comparison_results_df = pd.DataFrame(intermediate_results)
                comparison_results_df.to_csv("SIMPLE_COMPLEX.csv", index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f"An error occurred while processing {model_id}: {e}")

    if complex_samples_to_save:
        print("\n" + "="*50)
        print("Saving samples identified as 'COMPLEX' to CSV...")
        complex_df = pd.DataFrame(complex_samples_to_save)
        # The keys in the dictionary now directly match the desired CSV column headers,
        # so no renaming is necessary.
        output_path = "/kaggle/working/vimedaqa_complex.csv"
        complex_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Successfully saved {len(complex_df)} complex samples to: {output_path}")
        print("="*50)

else:
    print("Skipping generation due to issues with the dataset or Hugging Face token.")


# In[ ]:


# In Cell 5.5:

# --- Step 5.5 - Assemble and Save Results (Corrected Version) ---
if intermediate_results:
    print("\n" + "="*50)
    print("Assembling results into the final DataFrame...")
    print("="*50)
    
    # Convert the dictionary of results into a list of rows
    # final_results_list = list(intermediate_results.values())
    # results_df_hybrid = pd.DataFrame(final_results_list)
    results_df_hybrid = pd.DataFrame(intermediate_results)

    # --- START OF THE FIX ---
    # Define the exact and complete order of columns for the final CSV.
    
    # Start with the columns that should always be there.
    final_column_order = [
        "Sample_ID", "Model", "System_Prompt", 
        "Question", "Context", "Reasoning_Log", "Ground_Truth_Answer"
    ]
    
    # Dynamically find any remaining columns (like our answer column)
    # and append them. This is robust if you add more strategies later.
    answer_columns = [col for col in results_df_hybrid.columns if col not in final_column_order]
    final_column_order.extend(answer_columns)

    # Reorder the DataFrame, but only with columns that actually exist.
    # This prevents errors if a column is unexpectedly missing.
    existing_columns_in_order = [col for col in final_column_order if col in results_df_hybrid.columns]
    results_df_hybrid = results_df_hybrid[existing_columns_in_order]
    # --- END OF THE FIX ---

    # Sort the final DataFrame for consistency
    results_df_hybrid = results_df_hybrid.sort_values(by=["Sample_ID", "Model"]).reset_index(drop=True)

    output_file_path = "/kaggle/working/results.csv"
    results_df_hybrid.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    
    print(f"Complete! Saved {len(results_df_hybrid)} rows to the file:")
    print(output_file_path)
    
    display(results_df_hybrid.head())
    
else:
    print("\nNo results were generated to save.")


# # Load CSV to evaluate

# In[ ]:


# --- TOGGLE FOR RE-EVALUATION ---
# Set this to True to skip generation and load data directly from the specified CSV file.
# Set it to False to run the normal generation process.
RE_EVALUATE_FROM_CSV = False
RESULTS_CSV_PATH = "/kaggle/input/error-eval/SIMPLE_COMPLEX.csv" 

# This check prevents the loading code from running if you have just generated new answers.
# The 'all_generated_answers' dictionary will be empty unless the generation step was skipped.
if RE_EVALUATE_FROM_CSV:
    print("="*50)
    print(f"RE-EVALUATION MODE...")
    print("="*50)
    
    try:
        # Load the previously saved results into a DataFrame
        results_df = pd.read_csv(RESULTS_CSV_PATH)
        
        # --- Reconstruct the necessary data structures for the evaluation script ---
        
        # 1. Reconstruct `ground_truth_answers`
        # Ensure that NaN or empty values are treated as empty strings
        results_df['Ground_Truth_Answer'] = results_df['Ground_Truth_Answer'].fillna('')
        ground_truth_answers = [[str(ref)] for ref in results_df['Ground_Truth_Answer'].tolist()]

        # 2. Reconstruct `all_generated_answers` dictionary
        all_generated_answers = {}
        
        # Handle potential NaN values in answer columns before converting to list
        results_df['Answer_SIMPLE_Path'] = results_df['Answer_SIMPLE_Path'].fillna('SKIPPED')
        results_df['Answer_COMPLEX_Path'] = results_df['Answer_COMPLEX_Path'].fillna('SKIPPED')
        
        # This logic assumes the CSV may contain results from multiple models.
        # It groups by the 'Model' column to correctly build the dictionary.
        for model_id, group_df in results_df.groupby('Model'):
            print(f"  > Loading answers for model: {model_id}")
            
            # Define the keys for the dictionary, just like in the generation script
            simple_key = (model_id, "SIMPLE_PATH", "Concise_EN")
            complex_key = (model_id, "COMPLEX_PATH", "FollowUp_EN")
            
            # Populate the dictionary with the lists of answers from the CSV
            all_generated_answers[simple_key] = [str(ans) for ans in group_df['Answer_SIMPLE_Path'].tolist()]
            all_generated_answers[complex_key] = [str(ans) for ans in group_df['Answer_COMPLEX_Path'].tolist()]

        # 3. Reconstruct `generation_times` (with dummy data)
        generation_times = {}
        print("\nNOTE: Generation times are not available in the CSV. They will be reported as 0 in the evaluation results.")
        for key in all_generated_answers.keys():
            generation_times[key] = 0.0
            
        print(f"\nSuccessfully loaded {len(ground_truth_answers)} samples and prepared them for re-evaluation.")

    except FileNotFoundError:
        print(f"ERROR: The file '{RESULTS_CSV_PATH}' was not found. Cannot re-evaluate.")
        # Make sure this is empty so the evaluation script knows not to run
        all_generated_answers = {}
    except Exception as e:
        print(f"An error occurred while loading the CSV for re-evaluation: {e}")
        all_generated_answers = {}

# The existing evaluation code (Step 6) will now run using the data loaded from the CSV.
# It starts with 'if all_generated_answers:', which will be True if the loading was successful.


# In[ ]:


pd.set_option('display.max_rows', None)

# Step 6: Evaluate the Generated Answers (Modified to exclude "SKIPPED" samples)
if all_generated_answers:
    # Load all the metrics we need
    rouge_metric = evaluate.load('rouge')
    bleu_metric = evaluate.load('bleu')
    meteor_metric = evaluate.load('meteor')
    bertscore_metric = evaluate.load('bertscore')

    evaluation_results = []

    print("\n" + "="*50)
    print("Calculating Evaluation Metrics")
    print("="*50)

    for result_key, predictions in all_generated_answers.items():
        model_name, prompt_name, sp_name = result_key

        # --- NEW: Filtering logic to exclude "SKIPPED" samples ---
        filtered_predictions = []
        filtered_ground_truth = []

        # Iterate through predictions and ground truths together
        for pred, ref in zip(predictions, evaluation_ground_truths):
            # Only include the sample if the prediction is not "SKIPPED"
            if pred != "SKIPPED":
                filtered_predictions.append(pred)
                filtered_ground_truth.append(ref)
        
        # Report how many samples are being evaluated for this path
        print(f"  > Evaluating '{prompt_name}': Found {len(filtered_predictions)} non-SKIPPED samples out of {len(predictions)} total.")

        # --- Modified Check: Handle cases where all samples were skipped ---
        # If the filtered list is empty, it means all samples were "SKIPPED" or empty to begin with.
        if not filtered_predictions:
            print(f"  WARNING: No valid samples to evaluate for '{result_key}'. Assigning all metric scores to 0.")
            result_row = {
                "Model": model_name,
                "User_Prompt": prompt_name,
                "System_Prompt": sp_name,
                "ROUGE-L": 0.0,
                "BLEU": 0.0,
                "METEOR": 0.0,
                "BERTScore-F1": 0.0,
                "Avg-Score": 0.0,
                "Generation Time (s)": round(generation_times.get(result_key, 0), 2),
                "Evaluated_Samples": 0 # Add a column to track sample count
            }
            evaluation_results.append(result_row)
            continue # Move to the next result_key
    
        # --- Proceed with evaluation using the FILTERED lists ---
        rouge_scores = rouge_metric.compute(predictions=filtered_predictions, references=filtered_ground_truth)
        bleu_scores = bleu_metric.compute(predictions=filtered_predictions, references=filtered_ground_truth)
        meteor_scores = meteor_metric.compute(predictions=filtered_predictions, references=filtered_ground_truth)
        bertscore_scores = bertscore_metric.compute(predictions=filtered_predictions, references=filtered_ground_truth, lang="vi")
        
        # Calculate individual scores for the current result_key
        rouge_l = round(rouge_scores['rougeL'], 4)
        bleu = round(bleu_scores['bleu'], 4)
        meteor = round(meteor_scores['meteor'], 4)
        bertscore_f1 = round(sum(bertscore_scores['f1']) / len(bertscore_scores['f1']), 4)

        # Calculate the average score
        avg_score = round((rouge_l + bleu + meteor + bertscore_f1) / 4, 4)
    
        # Store results
        result_row = {
            "Model": model_name,
            "User_Prompt": prompt_name,
            "System_Prompt": sp_name,
            "ROUGE-L": rouge_l,
            "BLEU": bleu,
            "METEOR": meteor,
            "BERTScore-F1": bertscore_f1,
            "Avg-Score": avg_score,
            "Generation Time (s)": round(generation_times.get(result_key, 0), 2),
            "Evaluated_Samples": len(filtered_predictions) # Add a column to track sample count
        }
        evaluation_results.append(result_row)

    # Step 7: Display Results
    if evaluation_results:
        results_df = pd.DataFrame(evaluation_results)
        # Sort for better comparison
        results_df = results_df.sort_values(by="Avg-Score", ascending=False).reset_index(drop=True)
        print("\n--- Comparative Evaluation Results ---")
        display(results_df)

        # Save the evaluation results DataFrame to a separate CSV file
        evaluation_output_path = "/kaggle/working/eval_results.csv"
        results_df.to_csv(evaluation_output_path, index=False, encoding='utf-8-sig')
        print(f"\nEvaluation results successfully saved to: {evaluation_output_path}")

else:
    print("\nNo answers were generated. Skipping evaluation.")

