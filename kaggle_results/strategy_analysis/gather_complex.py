import pandas as pd
import os

# --- Configuration ---
# Set the path to the folder containing your CSV files.
# Use "." if the script is in the same folder as the CSVs.
folder_path = '.' 
# Set the name for the final output file.
output_filename = 'complex_samples_unique.csv'
# ---------------------

def consolidate_complex_samples(folder_path, output_filename):
    """
    Scans a folder for CSV files, extracts all rows where 'Chosen_Strategy' is 'COMPLEX',
    handles missing 'Model_Reason' column, removes duplicates, and saves to a new CSV.
    """
    all_complex_samples = []
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Get a list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f != output_filename]
    
    if not csv_files:
        print(f"No CSV files found in '{folder_path}'.")
        return

    print(f"Found {len(csv_files)} CSV files to process...")

    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {filename}")
        
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # --- Gracefully handle the optional 'Model_Reason' column ---
            if 'Model_Reason' not in df.columns:
                df['Model_Reason'] = 'N/A' # Add a placeholder if the column is missing

            # Ensure required columns exist before proceeding
            required_columns = ['Context', 'Question', 'Chosen_Strategy']
            if not all(col in df.columns for col in required_columns):
                print(f"  - Skipping '{filename}': Missing one of the required columns (Context, Question, Chosen_Strategy).")
                continue

            # --- Filter for rows where 'Chosen_Strategy' is COMPLEX ---
            complex_df = df[df['Chosen_Strategy'].str.upper() == 'COMPLEX'].copy()
            
            if not complex_df.empty:
                print(f"  - Found {len(complex_df)} 'COMPLEX' samples.")
                # Append the filtered DataFrame to our master list
                all_complex_samples.append(complex_df)
            else:
                print(f"  - No 'COMPLEX' samples found in this file.")

        except Exception as e:
            print(f"  - Could not process file '{filename}'. Error: {e}")

    # --- Combine, remove duplicates, and save ---
    if all_complex_samples:
        # Concatenate all the collected DataFrames into one
        final_df = pd.concat(all_complex_samples, ignore_index=True)
        
        print(f"\nTotal 'COMPLEX' samples collected: {len(final_df)}")

        # --- Remove duplicates based on both Context and Question ---
        # This ensures that each unique sample is kept only once.
        final_df_unique = final_df.drop_duplicates(subset=['Context', 'Question'], keep='first')
        
        print(f"Samples after removing duplicates: {len(final_df_unique)}")

        # Reorder columns for consistency
        final_df_unique = final_df_unique[['Context', 'Question', 'Chosen_Strategy', 'Model_Reason']]
        
        # Save the final, unique DataFrame to a new CSV file
        final_output_path = os.path.join(folder_path, output_filename)
        final_df_unique.to_csv(final_output_path, index=False, encoding='utf-8-sig')
        
        print(f"\nSuccessfully saved {len(final_df_unique)} unique 'COMPLEX' samples to '{final_output_path}'")
    else:
        print("\nNo 'COMPLEX' samples were found across all files. No output file was created.")

# --- Run the function ---
consolidate_complex_samples(folder_path, output_filename)