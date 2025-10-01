import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import PythonExporter
import os

def convert_ipynb_to_py(ipynb_file, py_file, include_markdown=True, execute_notebook=False):
    """
    Converts a Jupyter Notebook (.ipynb) to a Python script (.py).

    Args:
        ipynb_file (str): The path to the input .ipynb file.
        py_file (str): The path to the output .py file.
        include_markdown (bool): If True, markdown cells will be included as comments.
                                 If False, markdown cells will be excluded.
        execute_notebook (bool): If True, the notebook will be executed before conversion.
                                 The cell outputs themselves are not stored in the .py file,
                                 but the code that generates them will be present.
    """
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    if execute_notebook:
        # To execute the notebook, a preprocessor is needed.
        # This will run the notebook and update the output cells.
        # Note that for a .py file, the output itself is not saved,
        # but the code to generate it is.
        executor = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            executor.preprocess(nb, {'metadata': {'path': './'}})
        except Exception as e:
            print(f"Error executing the notebook: {e}")
            # Decide if you want to continue with the conversion despite the error
            # return

    # Configure the exporter
    exporter = PythonExporter()

    if not include_markdown:
        # To exclude markdown, we can use a preprocessor that removes markdown cells.
        # However, a simpler approach for direct Python export is to configure the exporter.
        # The PythonExporter by default includes markdown as comments.
        # A custom template or post-processing would be needed to fully remove them
        # in a library-native way without resorting to manual cell filtering.

        # A more direct approach is to filter the cells before exporting:
        code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
        nb.cells = code_cells

    # Export the notebook to a Python script
    (body, resources) = exporter.from_notebook_node(nb)

    # Write the Python script to a file
    with open(py_file, 'w', encoding='utf-8') as f:
        f.write(body)

    print(f"Successfully converted {ipynb_file} to {py_file}")

if __name__ == '__main__':
    # --- This is the updated section ---

    # 1. Define the name of the notebook you want to convert
    notebook_filename = 'unsloth'

    # 2. Get the absolute path to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 3. Create the full, absolute path to the input and output files
    input_notebook_path = os.path.join(script_dir, notebook_filename + '.ipynb')
    output_with_markdown = os.path.join(script_dir, notebook_filename + '.py')
    output_without_markdown = os.path.join(script_dir, notebook_filename + '_no_md.py')

    # Before running, make sure the input file actually exists
    if not os.path.exists(input_notebook_path):
        print(f"Error: Input file not found at '{input_notebook_path}'")
        print("Please make sure 'example.ipynb' is in the same folder as this script.")
    else:
        # --- Conversion Examples ---

        # # 1. Convert with markdown included
        # print("--- Converting with markdown ---")
        # convert_ipynb_to_py(input_notebook_path, output_with_markdown, include_markdown=True)

        # 2. Convert without markdown
        print("\n--- Converting without markdown ---")
        convert_ipynb_to_py(input_notebook_path, output_without_markdown, include_markdown=False)