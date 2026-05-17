# Phân Tích Kết Quả Dự Đoán Giá Cổ Phiếu — So Sánh Model

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
| Gemma4 E2B (run 1) | 16.82* | catastrophic | 35.92* | catastrophic | 2.76% | catastrophic |
| Phi 2.7B (run 0) | 173.68 | 195.68 | 278.57 | 312.40 | 25.09% | 37.89% |
| Phi 2.7B (run 1) | 214.53 | catastrophic | 318.56 | catastrophic | 37.84% | catastrophic |

*\*Gemma4 E2B loại trừ lb30*

### 3.2 Parse Fail Rate

| Model | Tổng Fail | Tổng Valid | Fail Rate |
|---|---|---|---|
| **Gemma3 4B (run 3.5)** | **0** | **2,150** | **0.00%** |
| Gemma4 E2B (run 1) | 103 | 972 | 9.59% |
| Phi 2.7B (run 0) | 703 | 1,361 | 34.07% |
| Phi 2.7B (run 1) | 298 | 1,897 | 13.59% |

---

## 4. Parse Fail Rate Theo Lookback

| Model | lb=1 | lb=7 | lb=14 | lb=21 | lb=30 |
|---|---|---|---|---|---|
| **Gemma3 4B (run 3.5)** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** |
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

# **Cần cải thiện:**
- Test thêm model mới (nemotron-mini 4b, phi4-mini:3.8b)
- Implement JSON mode để loại bỏ hoàn toàn parse fail
- Thêm few-shot examples vào prompt
- Tính technical indicators làm input features

### 1. Technical Indicators làm Input Features
Vấn đề hiện tại: Input là raw OHLCV (Open, High, Low, Close, Volume) → model phải tự nhận diện pattern từ số liệu thô. Với model nhỏ (2-4B), việc này quá khó.
Giải pháp: Tính sẵn các chỉ số kỹ thuật phổ biến:
- MA5/MA10/MA20 (Moving Average): Xu hướng giá ngắn/trung hạn
- RSI (Relative Strength Index): Overbought (>70) / Oversold (<30)
- Volatility (Standard Deviation): Độ biến động
- ATR (Average True Range): Biên độ dao động trung bình
Lợi ích:
- Input ngắn hơn (key ngắn: c, o, h, l, v, ma5, rsi...)
- Model dễ hiểu pattern hơn (ví dụ: RSI > 70 → giá có thể giảm)
- Giảm token count → ít bị overload ở lookback dài
Đã implement: compute_features() trong main_improved.py

### 2. Few-shot Examples trong Prompt
Vấn đề hiện tại: Prompt chỉ có ví dụ trừu tượng (number;number;...) → model không thấy input JSON thực tế trông như thế nào và output đúng phải ra sao.
Giải pháp: Thêm 1 ví dụ đầy đủ vào system prompt:
Input: {"symbol":"BAC","lb":7,"data":[{"c":45.12,"o":45.00,...},...]}
Output: 45.4200;45.5800;45.3100;...
Lợi ích:
- Model hiểu rõ format: "nhận JSON này → trả về dãy số kia"
- Giảm hallucination (model không output Python dict hay code nữa)
- Đặc biệt hiệu quả với model nhỏ (2-4B)
Đã implement: FEWSHOT_EXAMPLE trong main_improved.py

### 3. Compact Input Format
Vấn đề hiện tại: Input JSON dài dòng:
{"Date":"2025-12-01","open":45.1,"high":45.8,"low":44.9,"close":45.2,"volume":50000000}
Với lookback=30 → ~300-500 tokens chỉ cho input.
Giải pháp: Nén JSON:
- Bỏ Date (không cần thiết cho prediction)
- Dùng key ngắn: c, o, h, l, v thay vì close, open, high, low, volume
- Làm tròn số: 2 decimals thay vì full precision
Kết quả: Input ngắn hơn ~40-50% → model xử lý nhanh hơn, ít overload hơn.
Đã implement: prepare_input_json() + compute_features() trong main_improved.py

### 4. Smart Retry Mechanism
Vấn đề hiện tại: Retry chỉ đổi "4 decimals → 2 decimals" → không giải quyết gốc vấn đề. Model fail vì:
- Output Python dict: {'2021-12-31': {'open': 178.53, ...}}
- Output thiếu số: 25 thay vì 30
- Output JSON thay vì semicolon-separated
Giải pháp: Retry message nhấn mạnh đúng lỗi:
Attempt 1: "ERROR: DO NOT output Python dicts, JSON objects, or code. Output ONLY numbers separated by semicolons."
Attempt 2: "CRITICAL: You MUST output EXACTLY 30 numbers. Output ONLY: number;number;... (30 numbers total). NOTHING else."
Lợi ích: Retry giải quyết đúng nguyên nhân fail → tăng tỷ lệ thành công.
Đã implement: RETRY_MESSAGES trong main_improved.py

### 5. Ollama JSON Mode (CHƯA implement)
Vấn đề hiện tại: Parse text thủ công bằng regex → fragile, dễ fail.
Giải pháp: Dùng format parameter của Ollama để bắt buộc model output JSON đúng schema:
```
response = await client.chat(
    model=MODEL_NAME,
    messages=messages,
    format={
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": horizon,
                "maxItems": horizon
            }
        },
        "required": ["predictions"]
    }
)
```
Lợi ích:
- Ollama bắt buộc model output JSON đúng schema
- Không cần regex parse nữa → 0% parse fail do format sai
- Lấy trực tiếp response["predictions"]
Chưa implement vì cần test xem model có support JSON mode không.
