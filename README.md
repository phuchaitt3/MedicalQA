EDA Survey for:
1. https://github.com/tadeephuy/ViMQ/blob/main/data/test.json
2. https://arxiv.org/pdf/2501.18362
3. https://huggingface.co/datasets/tmnam20/ViMedAQA/viewer/all/train?row=0&views%5B%5D=all_train
4. https://github.com/tadeephuy/ViMQ/blob/main/data/test.json

Libraries:
- Remove Vietnamese filler words from vocab counting: pip install pyvi requests
- Before running download_dataset.py: pip install --upgrade datasets s3fs huggingface_hub pyarrow

Main analysis results in /eda_results for ViMQ and MedXpertQA datasets.

---
tui nghĩ bộ ViMedAQA khá tiềm năng
chỉ có 1 phần phải confirm là nó collect nhưng có check tay ko ?
nếu ko, thì mình có 2 goal hướng tới 1 - xây dựng hệ thống QA cho 2 bài toán yes/no và answer generation; 2 - đưa ra cảnh báo các câu sai ngữ cảnh từ model ở bước 1
nên 2 bạn build thử 1 model llms cho hướng 1 xem sao nhé.