import os
import random
import pandas as pd
from datasets import load_dataset, Dataset
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_similarity,
    answer_correctness,
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# from langchain.cache import BaseCache
ChatOpenAI.model_rebuild()

# Load environment variables from .env file
load_dotenv()

def load_vimedaqa_dataset(split="train"):
    """
    Loads the ViMedAQA dataset from Hugging Face.

    Args:
        split (str): The dataset split to load (e.g., "train").

    Returns:
        Dataset: The loaded Hugging Face dataset.
    """
    try:
        dataset = load_dataset("tmnam20/ViMedAQA", split=split)
        print("Dataset loaded successfully!")
        print("Example from the dataset:")
        print(dataset[0])
        return dataset
    except Exception as e:
        print(f"Failed to load the dataset. Error: {e}")
        return None

def get_openai_client():
    """
    Initializes and returns the OpenAI client using the API key
    from environment variables.

    Returns:
        OpenAI: The OpenAI client instance.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return OpenAI(api_key=api_key)

def format_prompt(sample):
    """
    Formats the prompt for the OpenAI model.

    Args:
        sample (dict): A sample from the dataset.

    Returns:
        str: The formatted prompt string.
    """
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

            Based on the context below, answer the question. 
            
            **Rules:**
            1. You MUST extract the answer directly from the context.
            2. The answer must be the exact, continuous text from the context.
            3. DO NOT add extra words or form a full sentence.
            
            **Example:**
            - Context: "Đến năm 1327, đây là thị trấn lớn thứ ba tại Warwickshire."
            - Question: "Vào thế kỉ XIV, Birmingham trở thành thị trấn lớn thứ mấy tại Warwickshire?"
            - Correct Answer: "lớn thứ ba"

            **Now, perform the task with the following:**
            
            Context: {sample['context']}
            
            Question: {sample['question']}

            <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def generate_openai_answer(client, prompt):
    """
    Generates an answer using the OpenAI API.

    Args:
        client (OpenAI): The OpenAI client.
        prompt (str): The prompt for the model.

    Returns:
        str: The generated answer.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred during OpenAI API call: {e}")
        return ""

def prepare_for_ragas(question, context, generated_answer, ground_truth):
    """
    Prepares the data in the format required for Ragas evaluation.

    Args:
        question (str): The question.
        context (str): The context.
        generated_answer (str): The model's generated answer.
        ground_truth (str): The ground truth answer.

    Returns:
        Dataset: A Hugging Face Dataset ready for Ragas.
    """
    ragas_data = {
        "question": [question],
        "contexts": [[context]],
        "answer": [generated_answer],
        "ground_truth": [ground_truth],
    }
    return Dataset.from_dict(ragas_data)

def run_ragas_evaluation(ragas_dataset):
    """
    Runs the Ragas evaluation sequentially for a set of metrics.

    Args:
        ragas_dataset (Dataset): The dataset prepared for Ragas.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation scores.
    """
    print("\nStarting Ragas evaluation SEQUENTIALLY to improve reliability...")
    print("=" * 50)

    evaluation_llm = ChatOpenAI(model="gpt-4.1-nano")
    metrics_to_run = [
        faithfulness,
        answer_relevancy,
        answer_similarity,
        answer_correctness,
    ]
    
    final_scores = {}
    for metric in metrics_to_run:
        metric_name = metric.name
        print(f"Evaluating metric: [{metric_name}]...")
        try:
            result = evaluate(dataset=ragas_dataset, metrics=[metric], llm=evaluation_llm)
            final_scores[metric_name] = result[metric_name]
        except Exception as e:
            print(f"  ERROR evaluating {metric_name}: {e}")
            final_scores[metric_name] = "Error"

    print("\n" + "=" * 50)
    print("Ragas sequential evaluation complete!")
    print("=" * 50 + "\n")

    return pd.DataFrame([final_scores])

if __name__ == "__main__":
    dataset = load_vimedaqa_dataset()
    if dataset:
        # Select a random sample for evaluation
        sample = dataset[random.randint(0, len(dataset) - 1)]
        
        try:
            openai_client = get_openai_client()
            
            # Generate the answer
            prompt = format_prompt(sample)
            generated_answer = generate_openai_answer(openai_client, prompt)
            
            # Prepare for Ragas
            ragas_dataset = prepare_for_ragas(
                sample["question"],
                sample["context"],
                generated_answer,
                sample["answer"],
            )

            print("\n--- Model Generation Complete ---")
            print(f"Sample ID: {sample['question_idx']}")
            print(f"Question: {sample['question']}")
            print(f"Model Answer: {generated_answer}")
            print(f"Ground Truth: {sample['answer']}")

            # Run Ragas evaluation
            results_df = run_ragas_evaluation(ragas_dataset)
            print("Ragas Evaluation Results:")
            print(results_df)

        except (ValueError, ImportError) as e:
            print(f"Error: {e}")