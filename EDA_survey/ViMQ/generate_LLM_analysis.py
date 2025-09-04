import google.generativeai as genai
import os
from dotenv import load_dotenv, dotenv_values

# --- Configuration ---
CUR_DIR = os.path.dirname(__file__)
dotenv_path = os.path.join(CUR_DIR, '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
api_key = os.getenv("GOOGLE_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL")
if not api_key or not gemini_model:
    print("ERROR: GOOGLE_API_KEY or GEMINI_MODEL environment variable not set.")
    print("Please set the environment variable or paste your key directly into the script.")
    exit()

genai.configure(api_key=api_key)

# Define file paths
EDA_RESULTS_DIR = os.path.join(CUR_DIR, "eda_results")
REPORT_FILE_PATH = os.path.join(EDA_RESULTS_DIR, "eda_report.txt")
ANALYSIS_OUTPUT_PATH = os.path.join(EDA_RESULTS_DIR, "gemini_analysis.md")

# --- Functions ---

def read_eda_report(file_path):
    """Reads the content of the EDA report file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: Report file not found at '{file_path}'")
        print("Please run the first EDA script to generate the report file.")
        return None

def build_prompt(report_content):
    """Constructs the full prompt for the Gemini API."""
    
    # Using a multiline f-string for a clean and readable prompt template
    prompt = f"""
    **CONTEXT:**
    I have performed an Exploratory Data Analysis (EDA) on the ViMQ (Vietnamese Medical Question) dataset. This dataset is used for two main NLP tasks: Intent Classification and Named Entity Recognition (NER). The dataset is split into training, development, and test sets.

    Here is the full EDA report:

    --- START OF REPORT ---
    {report_content}
    --- END OF REPORT ---

    **TASK:**
    As an expert data scientist, please analyze the EDA report above and provide the following in a well-structured Markdown format:

    1.  **Executive Summary:** A concise, high-level summary of the key characteristics of the ViMQ dataset.
    2.  **Key Insights & Observations:** Identify 3-5 critical insights from the data. Comment on the balance of intent labels, the complexity of sentences (length and entity count), the nature of the vocabulary, and any other notable patterns.
    3.  **Potential Challenges for Modeling:** Based on these insights, list the potential challenges a machine learning engineer might face when building an intent classification or NER model on this data (e.g., class imbalance, out-of-vocabulary words).
    4.  **Actionable Recommendations:** Suggest 3-4 concrete next steps for data preprocessing, model strategy, or further analysis before starting to train a machine learning model.
    """
    return prompt

def analyze_with_gemini(prompt):
    """Sends the prompt to the Gemini API and returns the response."""
    try:
        model = genai.GenerativeModel(gemini_model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        return None

def main():
    """Main function to run the analysis workflow."""
    print("--- Starting EDA Analysis with Gemini API ---")

    # 1. Read the report
    print(f"Reading EDA report from: {REPORT_FILE_PATH}")
    report_content = read_eda_report(REPORT_FILE_PATH)
    if not report_content:
        return

    # 2. Build the prompt
    print("Building the prompt for the Gemini API...")
    prompt = build_prompt(report_content)
    
    # 3. Call the Gemini API for analysis
    print("Sending request to Gemini... This may take a moment.")
    analysis_result = analyze_with_gemini(prompt)
    
    if analysis_result:
        # 4. Save the analysis to a file
        with open(ANALYSIS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write(analysis_result)
        print(f"\n--- Analysis Complete ---")
        print(f"Gemini's analysis has been saved to: {ANALYSIS_OUTPUT_PATH}")
    else:
        print("\n--- Analysis Failed ---")
        print("Could not retrieve a response from the Gemini API.")

if __name__ == "__main__":
    main()