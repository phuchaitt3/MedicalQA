# Báo cáo Phân tích Khám phá Dữ liệu (EDA) cho bộ dữ liệu ViMedAQA

## 1. Giới thiệu

Báo cáo này trình bày kết quả phân tích khám phá dữ liệu (EDA) cho bộ dữ liệu ViMedAQA, một tập dữ liệu hỏi đáp y tế tiếng Việt được thu thập từ YouMed.vn. Mục tiêu của EDA là hiểu rõ hơn về cấu trúc, nội dung và đặc điểm của dữ liệu, từ đó cung cấp cái nhìn sâu sắc cho các bước xử lý và mô hình hóa tiếp theo.

## 2. Thu thập và Cấu trúc Dữ liệu

Bộ dữ liệu ViMedAQA được tải từ Hugging Face. Nó bao gồm ba tập con: `train`, `test` và `validation`. Mỗi tập con chứa các trường sau:

- `question_idx`: ID của câu hỏi
- `question`: Nội dung câu hỏi
- `answer`: Nội dung câu trả lời
- `context`: Ngữ cảnh liên quan đến câu hỏi và câu trả lời
- `title`: Tiêu đề của bài viết gốc
- `keyword`: Từ khóa liên quan
- `topic`: Chủ đề của câu hỏi/câu trả lời
- `article_url`: URL của bài viết gốc
- `author`: Tác giả của bài viết
- `author_url`: URL của tác giả

Số lượng hàng trong mỗi tập con:
- `train`: 39881 hàng
- `test`: 2217 hàng
- `validation`: 2215 hàng

Tổng số hàng trong toàn bộ dữ liệu là 44313.

## 3. Thống kê Mô tả

### 3.1. Độ dài các trường văn bản

Chúng tôi đã tính toán độ dài (số ký tự) của các trường `question`, `answer` và `context` để hiểu về phân bố độ dài của chúng. Dưới đây là thống kê mô tả:

```
       question_len    answer_len   context_len
count  39881.000000  39881.000000  39881.000000
mean      61.197287    111.120809    520.612547
std       22.877678     64.110272    383.676244
min        9.000000      2.000000      0.000000
25%       45.000000     68.000000    251.000000
50%       58.000000     98.000000    429.000000
75%       74.000000    140.000000    688.000000
max      374.000000    967.000000   5712.000000
```

**Nhận xét:**
- Độ dài trung bình của câu hỏi là khoảng 61 ký tự, câu trả lời là 111 ký tự và ngữ cảnh là 520 ký tự.
- Có sự biến động đáng kể về độ dài, đặc biệt là ở trường `context` (độ lệch chuẩn cao và giá trị max lớn).
- Giá trị `min` của `context_len` là 0, cho thấy có những mẫu không có ngữ cảnh.

### 3.2. Số lượng giá trị duy nhất trong các cột phân loại

- Số lượng chủ đề duy nhất: 4
- Số lượng tác giả duy nhất: 144
- Số lượng từ khóa duy nhất: 1935

**Nhận xét:**
- Số lượng chủ đề khá ít, cho thấy dữ liệu có thể được phân loại thành các nhóm lớn.
- Có nhiều tác giả và từ khóa duy nhất, điều này có thể hữu ích cho việc phân tích sâu hơn về nguồn gốc và nội dung chuyên biệt.

### 3.3. Các giá trị hàng đầu trong các cột phân loại

**Top 5 chủ đề:**
```
topic
1    14121
3    12485
2     8802
0     4473
Name: count, dtype: int64
```

**Top 5 tác giả:**
```
author
Bác sĩ Phạm Lê Phương Mai       2895
Dược sĩ Trần Việt Linh          2767
Bác sĩ Phan Văn Giáo            1921
Dược sĩ Nguyễn Ngọc Cẩm Tiên    1919
ThS.BS Vũ Thành Đô              1629
Name: count, dtype: int64
```

**Top 5 từ khóa:**
```
keyword
Vú           79
Thận        74
Động mạch    64
Nước bọt     62
Cảm cúm      57
Name: count, dtype: int64
```

**Nhận xét:**
- Chủ đề 1 và 3 chiếm phần lớn dữ liệu.
- Một số tác giả đóng góp đáng kể vào bộ dữ liệu.
- Các từ khóa hàng đầu cho thấy các lĩnh vực y tế phổ biến trong bộ dữ liệu.

## 4. Trực quan hóa Dữ liệu

Các biểu đồ sau đây cung cấp cái nhìn trực quan về phân phối độ dài văn bản và tần suất của các chủ đề, tác giả, từ khóa.

### 4.1. Phân phối độ dài các trường văn bản

![Phân phối độ dài câu hỏi, câu trả lời và ngữ cảnh](https://private-us-east-1.manuscdn.com/sessionFile/QkdoOqmjJ8GClzHmjaZ1q1/sandbox/1gDOEFGYqz7Dtw0NzA27L7-images_1756914775032_na1fn_L2hvbWUvdWJ1bnR1L3RleHRfbGVuZ3Roc19kaXN0cmlidXRpb24.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUWtkb09xbWpKOEdDbHpIbWphWjFxMS9zYW5kYm94LzFnRE9FRkdZcXo3RHR3ME56QTI3TDctaW1hZ2VzXzE3NTY5MTQ3NzUwMzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzUmxlSFJmYkdWdVozUm9jMTlrYVhOMGNtbGlkWFJwYjI0LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=Sdo18JPyKP8C6iF9GrfGPyKnawp~yS~erufMs0SnmF14eaHIvgMbdwfxOsW5drN~tso~LlaPtN4uGkbA1a0Xmas78RfSv0r9GVHmiDpATYQyOliFZFRFd3i4TQ2NBhcLHoqgu~mE0qrD~YgP1uiMwhz4MafTZ-Azvmows~BMiqd7fuJiLzdh4gYisfHDf~KwxkSHcr4M9xi05vxrmZIJEtaQna9DGc2p4O0o2N6XLJA4--yUqv~ExQe0X5rJjSQbGteBblwF-sNoSf5I~SNThltDMJfoya8O9yRdg2s6uC6RgiUdecPFaXDnxZN~Rsi2~BnH0ALRSFI-lizI~5WZcg__)

### 4.2. Tần suất các chủ đề

![Tần suất các chủ đề hàng đầu](https://private-us-east-1.manuscdn.com/sessionFile/QkdoOqmjJ8GClzHmjaZ1q1/sandbox/1gDOEFGYqz7Dtw0NzA27L7-images_1756914775033_na1fn_L2hvbWUvdWJ1bnR1L3RvcF90b3BpY3NfZnJlcXVlbmN5.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUWtkb09xbWpKOEdDbHpIbWphWjFxMS9zYW5kYm94LzFnRE9FRkdZcXo3RHR3ME56QTI3TDctaW1hZ2VzXzE3NTY5MTQ3NzUwMzNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzUnZjRjkwYjNCcFkzTmZabkpsY1hWbGJtTjUucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=T1FyD0Wgkq8DK8Dq6XnBlPaDFSrpj~cI-4v1MPyY3ma9utit9y1VnfMk1FLQNMDARt79fSzmPflSVmE~e~mfSzJ1ZvQtGeFdVFCf2OmprNXL-Gv9JzoSyJIHnTn0mUnIeojmK3dkioFGGTJciXCR4MQpYsPoYUg0BMnn5iwZvPXSDdoaKNcspYdWt0NAofZLRH2intzp~Xi-3A4mb2KVEeYDF9VaQPWEnDxt2XJbVpydKKB~TyRffMaNpiUPmaOeIJwf9PS6q9JTFLgeq-8FTKTBp-hrG-6wrqNPIZsC5isRmFz9i8yRNGylN2h2HrfpydJALo14omq4wxYjQhjmJQ__)

### 4.3. Tần suất các tác giả hàng đầu

![Tần suất các tác giả hàng đầu](https://private-us-east-1.manuscdn.com/sessionFile/QkdoOqmjJ8GClzHmjaZ1q1/sandbox/1gDOEFGYqz7Dtw0NzA27L7-images_1756914775033_na1fn_L2hvbWUvdWJ1bnR1L3RvcF9hdXRob3JzX2ZyZXF1ZW5jeQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUWtkb09xbWpKOEdDbHpIbWphWjFxMS9zYW5kYm94LzFnRE9FRkdZcXo3RHR3ME56QTI3TDctaW1hZ2VzXzE3NTY5MTQ3NzUwMzNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzUnZjRjloZFhSb2IzSnpYMlp5WlhGMVpXNWplUS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=Hu4g9CDf91iB9dCzRw3fN-QYT9oZFgQDpC0EgyURUmTlU7dIRobFlWxBHoyAVQt4HbaLwQVafxAI7crT3~RDkL9kqypvSg17bKDhqvmvdrQm7-RpfDfJVnaaKPxyc9-DE~BTrLEt13gTV~OZHqjmXfwuDr0C~J~x3kC5ZudEqMsLd71U78L8R5jNVVTvFfF~eovI659a6HMO5MC-rJvnVEH92waBe1My98AwQGuUffVoTRm6kgSun-Fh1rmAO0R~PxttWphuNW37ER3H22CrwlJzYwHizD7WnI3RJN4z2Br0Rk1zcedEAKQoSFkNQyNYITLAoBL4fHII4b22J8gdrw__)

### 4.4. Tần suất các từ khóa hàng đầu

![Tần suất các từ khóa hàng đầu](https://private-us-east-1.manuscdn.com/sessionFile/QkdoOqmjJ8GClzHmjaZ1q1/sandbox/1gDOEFGYqz7Dtw0NzA27L7-images_1756914775033_na1fn_L2hvbWUvdWJ1bnR1L3RvcF9rZXl3b3Jkc19mcmVxdWVuY3k.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUWtkb09xbWpKOEdDbHpIbWphWjFxMS9zYW5kYm94LzFnRE9FRkdZcXo3RHR3ME56QTI3TDctaW1hZ2VzXzE3NTY5MTQ3NzUwMzNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzUnZjRjlyWlhsM2IzSmtjMTltY21WeGRXVnVZM2sucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=VaIBYKzxipsn3HC~W10x8FEhx~wnTMtk-7tK66GrdXXxGTSUnz1cBRVg-x-UKJr4cqffPIiKlFVOEyV1gMrzoFiqu0jgmxAVvwW5Bdoxdfc-NM0sPYa3Obvex~9Y1-uVSr2Oom3UIYmg7oY4xa6VSPtRkySLZJoOFwaqtHMQ1kCFFFmiXO6Fz9B-cYQCtOlQjgXuSShX1xCz0xPRb6L-CcopGtVijjIgfJiSE9Mvi3ifybEY6OQmsyHwKO4s4U0AUu8DE59Ck-EnEaWlQhTc6uWEQk9URZl3G20FIFJywW3TGUM6yvrIDmghi~LECPCcTQGV~NsBjks0WjXNm7r3ww__)

## 5. Kết luận

Phân tích EDA đã cung cấp những hiểu biết quan trọng về bộ dữ liệu ViMedAQA. Chúng tôi đã khám phá cấu trúc dữ liệu, phân tích thống kê mô tả về độ dài văn bản và phân bố các trường phân loại như chủ đề, tác giả và từ khóa. Các biểu đồ trực quan hóa đã giúp làm rõ hơn những đặc điểm này.

Những phát hiện này sẽ là cơ sở quan trọng cho các bước tiếp theo trong việc phát triển mô hình xử lý ngôn ngữ tự nhiên trên bộ dữ liệu này, bao gồm tiền xử lý dữ liệu, lựa chọn tính năng và thiết kế kiến trúc mô hình.

