from datasets import load_dataset

# To load the text subset of MedXpertQA
medxpertqa_text = load_dataset("TsinghuaC3I/MedXpertQA", "Text")

# To load the multimodal (mm) subset of MedXpertQA
medxpertqa_mm = load_dataset("TsinghuaC3I/MedXpertQA", "MM")

# You can then access different splits, e.g., the test set
test_data_text = medxpertqa_text['test']
test_data_mm = medxpertqa_mm['test']

# You can also save it locally in various formats if needed
test_data_text.to_json("medxpertqa_text_test.jsonl")
test_data_mm.to_json("medxpertqa_mm_test.jsonl")