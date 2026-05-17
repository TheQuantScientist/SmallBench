# Phân Tích Kết Quả Dự Đoán Giá Cổ Phiếu — So Sánh Model

> **Ngày phân tích:** 2026-05-17
> **Dataset:** 10 cổ phiếu tài chính (JPM, BAC, WFC, C, GS, MS, V, MA, AXP, BLK)
> **Test period:** 2026-01-01 → 2026-04-17
> **Forecast horizon:** 30 ngày
> **Lookbacks:** 1, 7, 14, 21, 30 ngày

---

## 1. Tổng Quan Các Model Đã Test

| Model | Size | Folder | Ghi chú |
|---|---|---|---|
| Gemma3 4B (run 0) | 4B | `result_gemma3_4b/` | Có `fh28` và `fh30` lẫn lộn |
| Gemma3 4B (run 1) | 4B | `result_gemma3_4b_1/` | Prompt chưa tối ưu |
| Gemma3 4B (run 2) | 4B | `result_gemma3_4b_2/` | Tương tự run 1 |
| **Gemma3 4B (run 3.5)** | **4B** | **`result_gemma3_4b_3_5/`** | **Prompt đã fix — 0% fail** |
| Gemma4 E2B (run 1) | 2B | `result_gemma4_e2b_1/` | Chỉ test 5/10 symbols, lb30 catastrophic |
| Phi 2.7B (run 0) | 2.7B | `result_phi_2.7b/` | Fail rate rất cao |
| Phi 2.7B (run 1) | 2.7B | `result_phi_2.7b_1/` | Numerical explosion ở step 30 |

---

## 2. Bảng Tổng Hợp — MAE / RMSE / MAPE / Fail Rate Theo Lookback

### 2.1 Gemma3 4B (run 3.5) — **BEST MODEL**

| Lookback | Valid | Fail | Fail Rate | MAE@1 | MAE@30 | RMSE@1 | RMSE@30 | MAPE@1 | MAPE@30 |
|---|---|---|---|---|---|---|---|---|---|
| lb=1 | 430 | 0 | **0.00%** | 6.28 | 47.89 | 8.58 | 53.80 | 1.60% | 13.25% |
| lb=7 | 430 | 0 | **0.00%** | 8.41 | 59.84 | 11.04 | 70.85 | 2.33% | 16.95% |
| lb=14 | 430 | 0 | **0.00%** | 12.45 | 33.09 | 15.62 | 39.00 | 3.35% | 9.13% |
| lb=21 | 430 | 0 | **0.00%** | 11.06 | 36.09 | 14.20 | 42.66 | 2.84% | 10.06% |
| lb=30 | 430 | 0 | **0.00%** | 14.78 | 35.02 | 19.07 | 41.06 | 3.37% | 10.02% |
| **Trung bình** | **430** | **0** | **0.00%** | **10.60** | **42.39** | **13.70** | **49.47** | **2.70%** | **11.88%** |

> **Nhận xét:** 0 parse fail ở mọi lookback. MAE@30 tốt nhất (35-47). lb=14 cho MAE@30 tốt nhất (33.09).

---

### 2.2 Gemma3 4B (run 0)

| Lookback | Valid | Fail | Fail Rate | MAE@1 | MAE@30 | RMSE@1 | RMSE@30 | MAPE@1 | MAPE@30 |
|---|---|---|---|---|---|---|---|---|---|
| lb=1 | 428 | 6 | 1.38% | 6.20 | 50.39 | 8.51 | 56.32 | 1.60% | 12.48% |
| lb=7 | 432 | 2 | 0.46% | 7.70 | 68.11 | 10.33 | 77.58 | 2.11% | 16.83% |
| lb=14 | 432 | 2 | 0.46% | 10.11 | 44.17 | 13.09 | 53.68 | 2.76% | 10.94% |
| lb=21 | 434 | 0 | 0.00% | 9.28 | 47.14 | 12.28 | 56.96 | 2.51% | 11.70% |
| lb=30 | 344 | 0 | 0.00% | 11.72 | 46.45 | 15.32 | 56.40 | 2.67% | 11.45% |
| **Trung bình** | **414** | **2** | **0.46%** | **9.00** | **51.25** | **11.91** | **60.19** | **2.33%** | **12.68%** |

---

### 2.3 Gemma3 4B (run 1)

| Lookback | Valid | Fail | Fail Rate | MAE@1 | MAE@30 | RMSE@1 | RMSE@30 | MAPE@1 | MAPE@30 |
|---|---|---|---|---|---|---|---|---|---|
| lb=1 | 424 | 6 | 1.40% | 6.29 | 48.07 | 8.59 | 53.79 | 1.60% | 13.39% |
| lb=7 | 414 | 16 | 3.72% | 8.46 | 60.11 | 11.06 | 71.32 | 2.31% | 16.73% |
| lb=14 | 381 | 49 | 11.40% | 12.56 | 37.16 | 15.68 | 52.16 | 3.37% | 9.55% |
| lb=21 | 344 | 86 | 20.00% | 11.52 | 34.82 | 14.61 | 41.15 | 2.92% | 10.00% |
| lb=30 | 309 | 121 | 28.14% | 15.80 | 34.65 | 20.38 | 40.93 | 3.33% | 10.16% |
| **Trung bình** | **374** | **56** | **12.93%** | **10.93** | **42.96** | **14.06** | **51.87** | **2.71%** | **11.97%** |

> **Nhận xét:** Fail rate tăng mạnh theo lookback (1.4% → 28%). Metrics khi valid vẫn khá tốt.

---

### 2.4 Gemma3 4B (run 2)

| Lookback | Valid | Fail | Fail Rate | MAE@1 | MAE@30 | RMSE@1 | RMSE@30 | MAPE@1 | MAPE@30 |
|---|---|---|---|---|---|---|---|---|---|
| lb=1 | 423 | 7 | 1.63% | 6.23 | 47.79 | 8.47 | 53.47 | 1.60% | 13.31% |
| lb=7 | 410 | 20 | 4.65% | 8.46 | 60.66 | 11.07 | 71.10 | 2.31% | 16.83% |
| lb=14 | 378 | 52 | 12.09% | 12.50 | 34.34 | 15.67 | 45.95 | 3.39% | 9.28% |
| lb=21 | 344 | 86 | 20.00% | 11.38 | 35.00 | 14.44 | 41.97 | 2.90% | 10.01% |
| lb=30 | 308 | 122 | 28.37% | 16.10 | 35.04 | 20.57 | 41.37 | 3.34% | 10.31% |
| **Trung bình** | **373** | **57** | **13.35%** | **10.93** | **42.57** | **14.04** | **50.77** | **2.71%** | **11.95%** |

---

### 2.5 Gemma4 E2B (run 1)

| Lookback | Valid | Fail | Fail Rate | MAE@1 | MAE@30 | RMSE@1 | RMSE@30 | MAPE@1 | MAPE@30 |
|---|---|---|---|---|---|---|---|---|---|
| lb=1 | 211 | 4 | 1.86% | 7.72 | 42.54 | 10.49 | 49.40 | 1.58% | 9.23% |
| lb=7 | 215 | 0 | 0.00% | 8.15 | 52.05 | 11.10 | 59.91 | 1.64% | 11.29% |
| lb=14 | 215 | 0 | 0.00% | 7.97 | 53.72 | 10.91 | 61.45 | 1.62% | 11.37% |
| lb=21 | 214 | 1 | 0.47% | 8.04 | 117.67 | 11.06 | 453.73 | 1.63% | 64.36% |
| lb=30 | 117 | 98 | **45.58%** | 52.24 | **72,428,564,985** | 135.92 | **338,099,201,907** | 7.34% | **65,327,164,129%** |
| **Trung bình** | **194** | **21** | **9.59%** | **16.82** | **catastrophic** | **35.92** | **catastrophic** | **2.76%** | **catastrophic** |

> **Nhận xét:** lb=30 **catastrophic failure** — numerical explosion ra hàng tỷ. Chỉ test 5/10 symbols (AXP, BAC, BLK, C, GS). Không ổn định.

---

### 2.6 Phi 2.7B (run 0)

| Lookback | Valid | Fail | Fail Rate | MAE@1 | MAE@30 | RMSE@1 | RMSE@30 | MAPE@1 | MAPE@30 |
|---|---|---|---|---|---|---|---|---|---|
| lb=1 | 356 | 74 | 17.21% | 257.92 | 263.12 | 376.28 | 403.75 | 39.86% | 51.66% |
| lb=7 | 363 | 67 | 15.58% | 248.81 | 274.92 | 460.95 | 499.82 | 25.88% | 39.60% |
| lb=14 | 281 | 149 | 34.65% | 168.60 | 213.60 | 337.40 | 392.70 | 22.89% | 38.63% |
| lb=21 | 238 | 192 | 44.65% | 174.94 | 179.09 | 190.95 | 191.17 | 28.97% | 39.29% |
| lb=30 | 123 | 221 | 64.24% | 18.14 | 47.65 | 27.25 | 74.57 | 7.84% | 20.28% |
| **Trung bình** | **272** | **141** | **34.07%** | **173.68** | **195.68** | **278.57** | **312.40** | **25.09%** | **37.89%** |

> **Nhận xét:** MAE@1 = 173 (tệ gấp 27x Gemma3). Output thường là Python dict thay vì số. Fail rate 34%.

---

### 2.7 Phi 2.7B (run 1)

| Lookback | Valid | Fail | Fail Rate | MAE@1 | MAE@30 | RMSE@1 | RMSE@30 | MAPE@1 | MAPE@30 |
|---|---|---|---|---|---|---|---|---|---|
| lb=1 | 459 | 16 | 3.37% | 278.65 | **657,381** | 342.51 | **4,263,331** | 57.09% | **345,690%** |
| lb=7 | 415 | 15 | 3.49% | 186.32 | **16,062,120,561** | 372.34 | **89,277,306,447** | 25.56% | **15,436,289,015%** |
| lb=14 | 428 | 2 | 0.47% | 181.66 | **1,515,015,918,879** | 373.36 | **9,755,792,184,334** | 23.66% | **2,906,473,994,469%** |
| lb=21 | 428 | 2 | 0.47% | 258.27 | **170,313,104,191** | 268.75 | **1,103,755,064,081** | 58.51% | **325,926,903,991%** |
| lb=30 | 167 | 263 | 61.16% | 167.73 | **819,393,815,499** | 235.86 | **2,939,583,745,238** | 24.37% | **268,040,484,442%** |
| **Trung bình** | **379** | **60** | **13.59%** | **214.53** | **catastrophic** | **318.56** | **catastrophic** | **37.84%** | **catastrophic** |

> **Nhận xét:** **CATASTROPHIC** — numerical explosion ở mọi lookback step 30. Model output số vô lý (hàng tỷ, hàng nghìn tỷ). Không dùng được.

---

## 3. So Sánh Trực Tiếp — Trung Bình Tất Cả Lookbacks

### 3.1 Metrics

| Model | MAE@1 | MAE@30 | RMSE@1 | RMSE@30 | MAPE@1 | MAPE@30 |
|---|---|---|---|---|---|---|
| **Gemma3 4B (run 3.5)** | **10.60** | **42.39** | **13.70** | **49.47** | **2.70%** | **11.88%** |
| Gemma3 4B (run 0) | 9.00 | 51.25 | 11.91 | 60.19 | 2.33% | 12.68% |
| Gemma3 4B (run 1) | 10.93 | 42.96 | 14.06 | 51.87 | 2.71% | 11.97% |
| Gemma3 4B (run 2) | 10.93 | 42.57 | 14.04 | 50.77 | 2.71% | 11.95% |
| Gemma4 E2B (run 1) | 16.82* | catastrophic | 35.92* | catastrophic | 2.76% | catastrophic |
| Phi 2.7B (run 0) | 173.68 | 195.68 | 278.57 | 312.40 | 25.09% | 37.89% |
| Phi 2.7B (run 1) | 214.53 | catastrophic | 318.56 | catastrophic | 37.84% | catastrophic |

*\*Gemma4 E2B loại trừ lb30*

### 3.2 Parse Fail Rate

| Model | Tổng Fail | Tổng Valid | Fail Rate |
|---|---|---|---|
| **Gemma3 4B (run 3.5)** | **0** | **2,150** | **0.00%** |
| Gemma3 4B (run 0) | 10 | 2,070 | 0.48% |
| Gemma3 4B (run 1) | 278 | 1,872 | 12.93% |
| Gemma3 4B (run 2) | 287 | 1,863 | 13.35% |
| Gemma4 E2B (run 1) | 103 | 972 | 9.59% |
| Phi 2.7B (run 0) | 703 | 1,361 | 34.07% |
| Phi 2.7B (run 1) | 298 | 1,897 | 13.59% |

---

## 4. Parse Fail Rate Theo Lookback

| Model | lb=1 | lb=7 | lb=14 | lb=21 | lb=30 |
|---|---|---|---|---|---|
| **Gemma3 4B (run 3.5)** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** |
| Gemma3 4B (run 0) | 1.38% | 0.46% | 0.46% | 0.00% | 0.00% |
| Gemma3 4B (run 1) | 1.40% | 3.72% | 11.40% | 20.00% | 28.14% |
| Gemma3 4B (run 2) | 1.63% | 4.65% | 12.09% | 20.00% | 28.37% |
| Gemma4 E2B (run 1) | 1.86% | 0.00% | 0.00% | 0.47% | **45.58%** |
| Phi 2.7B (run 0) | 17.21% | 15.58% | 34.65% | 44.65% | **64.24%** |
| Phi 2.7B (run 1) | 3.37% | 3.49% | 0.47% | 0.47% | **61.16%** |

> **Xu hướng rõ ràng:** Lookback càng dài → fail rate càng cao (trừ run 3.5 đã fix prompt).

---

## 5. Nhận Xét Chính

### 5.1 Model Tốt Nhất: Gemma3 4B (run 3.5)

- **0% parse fail** ở mọi lookback — duy nhất model đạt được
- **MAE@30 thấp nhất** (42.39 trung bình)
- **MAPE@30 thấp nhất** (11.88%)
- Ổn nhất qua tất cả lookbacks
- **Key difference:** Prompt đã được fix giữa run 2 và run 3.5

### 5.2 Gemma3 — Cải Thiện Qua Các Run

| Run | Prompt | Fail Rate | Ghi chú |
|---|---|---|---|
| run 0 | Chưa rõ | 0.48% | Có thể đã có fix nhẹ |
| run 1 | Base template | 12.93% | Fail tăng theo lookback |
| run 2 | Tương tự run 1 | 13.35% | Kết quả tương đương run 1 |
| **run 3.5** | **Đã fix** | **0.00%** | **Thay đổi đáng kể** |

### 5.3 Gemma4 E2B — Không Ổn Định

- lb=1 đến lb=14 khá tốt (MAE@1 ~8, MAPE@1 ~1.6%)
- **lb=21 bắt đầu có vấn đề** (MAE@30 = 117, MAPE@30 = 64%)
- **lb=30 catastrophic** — numerical explosion (MAE = 72 tỷ)
- Chỉ test 5/10 symbols → dữ liệu không đầy đủ
- **Không nên dùng cho production**

### 5.4 Phi 2.7B — Không Dùng Được

- **MAE@1 = 173-215** (tệ gấp 16-20x Gemma3)
- **MAPE@1 = 25-57%** (tệ gấp 10-20x Gemma3)
- Output thường là Python dict, JSON, hoặc code thay vì số
- **Run 1: numerical explosion** ở mọi lookback step 30
- Model quá nhỏ cho task này

---

## 6. Nguyên Nhân Parse Fail

### 6.1 Gemma3 (run 1, 2)

| Lookback | Nguyên nhân chính |
|---|---|
| lb=1-7 | Model output thiếu số (25-29 thay vì 30) |
| lb=14-21 | Model thêm text thừa: stock symbol, giải thích |
| lb=30 | Model output JSON/dict thay vì semicolon-separated numbers |

### 6.2 Gemma4 E2B

| Lookback | Nguyên nhân chính |
|---|---|
| lb=1-7 | Model output thiếu số |
| lb=21 | Bắt đầu hallucinate — output số vô lý |
| lb=30 | **Numerical explosion** — model sinh số hàng tỷ, không kiểm soát được |

### 6.3 Phi 2.7B

| Lookback | Nguyên nhân chính |
|---|---|
| Tất cả | **Output Python dict/JSON** thay vì số |
| lb=30 | Attention overload — input quá dài, model quên instruction |
| Run 1 | **Numerical explosion** — sinh số cực lớn ở step 14-30 |

---

## 7. Khuyến Nghị

### 7.1 Model

| Ưu tiên | Model | Lý do |
|---|---|---|
| **1** | **Gemma3 4B (run 3.5 prompt)** | Tốt nhất hiện tại, 0% fail |
| **2** | **qwen2.5:3b** | Follow instruction rất tốt, đáng thử |
| **3** | **phi4-mini:3.8b** | Mạnh hơn phi:2.7b rất nhiều |
| **4** | **llama3.2:3b** | Ổn định, community support tốt |
| **KHÔNG** | phi:2.7b | Quá nhỏ, fail 34%, MAE tệ 20x |
| **KHÔNG** | gemma4:e2b | Unstable, lb30 catastrophic |

### 7.2 Prompt

1. **Thêm few-shot examples** — input JSON thực tế → output đúng
2. **Dùng JSON mode** của Ollama (`format` parameter) thay vì parse text
3. **Retry mechanism** phải nhấn mạnh đúng lỗi, không chỉ đổi số decimals

### 7.3 Input Data

1. **Nén input** — truyền compact format thay vì JSON dài
2. **Tính sẵn features** — MA, RSI, volatility thay vì raw OHLCV
3. **Giảm lookback tối đa** — lb=14 cho MAE@30 tốt nhất với Gemma3

---

## 8. Kết Luận

**Gemma3 4B với prompt của run 3.5 là lựa chọn tốt nhất hiện tại.**
- 0% parse fail, MAE@30 = 42.39, MAPE@30 = 11.88%
- Ổn định qua mọi lookback
- Cần tiếp tục dùng prompt này làm baseline

**Cần cải thiện:**
- Test thêm model mới (qwen2.5:3b, phi4-mini:3.8b)
- Implement JSON mode để loại bỏ hoàn toàn parse fail
- Thêm few-shot examples vào prompt
- Tính technical indicators làm input features
