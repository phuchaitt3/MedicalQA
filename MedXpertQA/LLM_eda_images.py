import openai
import os
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
import io
import base64

def pil_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 encoded string, handling transparency."""
    buffered = io.BytesIO()
    
    # Check if the image has an alpha channel (transparency)
    if image.mode in ('RGBA', 'P'):
        # Convert to RGB, which JPEG supports.
        # This will add a white background to the transparent parts.
        image = image.convert('RGB')
        
    # OpenAI vision models generally prefer JPEG for efficiency
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_eda_report_with_openai():
    """
    Parses the EDA report, loads text and associated images,
    sends them to the specified OpenAI model for analysis,
    and saves the result to a markdown file.
    """
    # --- 1. Configuration from .env ---
    print("Configuring API and setting up paths...")
    try:
        # Get the directory of the current script.
        # This makes the script runnable from anywhere.
        try:
            # This works when running as a .py file
            CUR_DIR = Path(__file__).parent
        except NameError:
            # This provides a fallback for interactive environments like Jupyter
            CUR_DIR = Path.cwd()

        # Load environment variables from a .env file located one directory up
        dotenv_path = CUR_DIR.parent / '.env'
        print(dotenv_path)
        load_dotenv(dotenv_path=dotenv_path)
        
        api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL")

        if not api_key or not openai_model:
            raise ValueError("OPENAI_API_KEY or OPENAI_MODEL not set in .env file.")

        openai.api_key = api_key
        print(f"API configured successfully. Using model: {openai_model}")

        # --- 2. Define File Paths ---
        EDA_RESULTS_DIR = CUR_DIR / "eda_results"
        REPORT_FILE_PATH = EDA_RESULTS_DIR / "eda_report.txt"
        ANALYSIS_OUTPUT_PATH = EDA_RESULTS_DIR / "openai_analysis.md"

        if not REPORT_FILE_PATH.is_file():
            print(f"Error: Report file not found at {REPORT_FILE_PATH}")
            print("Please run the EDA notebook first to generate the report.")
            return

    except Exception as e:
        print(f"Error during configuration: {e}")
        return

    # --- 3. Parse the Report and Construct the Prompt ---
    print(f"Reading report from: {REPORT_FILE_PATH}")
    openai_content_parts = [] # This list will hold the content for the OpenAI 'user' message
    
    analysis_prompt = """
    You are a senior data analyst specializing in medical data science. 
    I have performed an Exploratory Data Analysis (EDA) on the MedXpertQA dataset. 
    Below is my full report, which includes statistical summaries, data tables, and the visual plots that were generated.

    Please provide a comprehensive analysis and summary of my findings based on ALL the information provided (text, tables, and images). I want you to:
    1.  Provide a high-level executive summary of the dataset's characteristics.
    2.  Highlight the most important insights from the categorical, numerical, and vocabulary analyses.
    3.  Analyze the relationships revealed in the heatmaps and multimodal plots. Do these correlations make sense in a medical context?
    4.  Identify any potential data quality issues, biases, or limitations that this EDA reveals.
    5.  Suggest 2-3 next steps for further analysis or for preparing this data for a machine learning model.

    Synthesize information from both the text/tables and the images to form your conclusions. Structure your response in clear, well-formatted Markdown.
    """
    # Add the initial analysis prompt as the first text part for OpenAI
    openai_content_parts.append({"type": "text", "text": analysis_prompt})

    try:
        with open(REPORT_FILE_PATH, 'r', encoding='utf-8') as f:
            current_text_block = ""
            for line in f:
                if "Image saved to:" in line:
                    if current_text_block:
                        openai_content_parts.append({"type": "text", "text": current_text_block.strip()})
                    
                    current_text_block = ""

                    try:
                        # Extract the absolute path saved in the report
                        image_path_str = line.split("Image saved to:")[1].strip()
                        image_path = Path(image_path_str)

                        if image_path.is_file():
                            print(f"Found and adding image: {image_path.name}")
                            img = Image.open(image_path)
                            base64_image = pil_to_base64(img)
                            openai_content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            })
                        else:
                            print(f"Warning: Image file not found at '{image_path_str}', skipping.")
                            openai_content_parts.append({"type": "text", "text": f"\n[Note: Image at path '{image_path_str}' was not found.]\n"})
                    except IndexError:
                        current_text_block += line
                else:
                    current_text_block += line
            
            if current_text_block:
                openai_content_parts.append({"type": "text", "text": current_text_block.strip()})
        
        # Final OpenAI messages structure
        openai_messages = [{"role": "user", "content": openai_content_parts}]

    except Exception as e:
        print(f"Error while reading or parsing the report file: {e}")
        return

    # --- 4. Call the Gemini API ---
    try:
        print("\nSending report and images to OpenAI for analysis... (This may take a moment)")
        
        # OpenAI API call for chat completions with streaming
        stream_response = openai.chat.completions.create(
            model=openai_model, # Use the configured OpenAI model
            messages=openai_messages, # Use the constructed multimodal messages
            stream=True,
            max_tokens=4096 # Good practice to set a max_tokens for vision models
        )

        # --- 5. Save the Analysis to a File ---
        print(f"Saving OpenAI's analysis to: {ANALYSIS_OUTPUT_PATH}")
        full_response_content = ""
        for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response_content += chunk.choices[0].delta.content

        with open(ANALYSIS_OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(full_response_content) # Write the accumulated content

        print("Analysis complete and saved successfully.")

    except Exception as e:
        print(f"\nAn error occurred during the API call: {e}")
        print("Please check your API key, internet connection, and ensure you are using an OpenAI model capable of multimodal input (e.g., 'gpt-4o', 'gpt-4-vision-preview').")

if __name__ == "__main__":
    analyze_eda_report_with_openai()