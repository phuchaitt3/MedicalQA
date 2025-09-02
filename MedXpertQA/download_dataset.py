from datasets import load_dataset
import json

# To load the text subset of MedXpertQA
medxpertqa_text = load_dataset("TsinghuaC3I/MedXpertQA", "Text")

# To load the multimodal (mm) subset of MedXpertQA
medxpertqa_mm = load_dataset("TsinghuaC3I/MedXpertQA", "MM")

# You can then access different splits, e.g., the test set
test_data_text = medxpertqa_text['test']
test_data_mm = medxpertqa_mm['test']

# Convert the dataset split to a list of dictionaries
text_data_list = [example for example in test_data_text]
mm_data_list = [example for example in test_data_mm]

# Save the list to a JSON file
with open("medxpertqa_test.json", "w", encoding="utf-8") as f:
    json.dump(text_data_list, f, indent=4, ensure_ascii=False)

with open("medxpertqa_mm_test.json", "w", encoding="utf-8") as f:
    json.dump(mm_data_list, f, indent=4, ensure_ascii=False)