## Docs: update 1
Phân tích parse fail trong result_gemma4_e2b_1:
|Lookback | Tổng | Fail|
|---------|------|-----|
|lb7	  |43	 |  10 |
|lb14	  | 43	 |  17 |

### Nguyên nhân chính: Model trả về thiếu số (25-29 số thay vì 30). Ví dụ:
- 351.0602;351.0603;...;351.0626 → chỉ 25 số
- 358.53;358.53;...;358.53 → chỉ 25 số
Nguyên nhân phụ: Model không tuân thủ đúng số lượng yêu cầu, có thể do num_predict=1500 chưa đủ hoặc model bị chặn giới hạn token.
Để giảm fail rate: Cần tăng num_predict hoặc thêm logic retry với prompt nhắc model output đủ số.

### ==> main_gemma3.py fix thành công 0 parse fail
---
## Docs: update 10_05_26

Chạy cùng source với phi:2.7b và gemma4:e2b thì kết quả phi như sau:
![alt text](table_phi_2.7.png)

Tỷ lệ khá ổn, nhưng với lookback 30 thì vẫn còn khá cao

Phân tích nguyên nhân parse fail cao ở model phi:2.7b:
1. Model quá nhỏ cho task này
Phi-2 (2.7B parameters) là model rất nhỏ, khó làm theo instruction phức tạp về format output.
2. Model bắt chước format input thay vì output instruction
Input là JSON dict/array → model có xu hướng tiếp tục xuất dict giống input thay vì output dạng số;số;số như yêu cầu. Bằng chứng là raw_output trong failures chứa đầy:
- {'2021-12-31': {'open': 178.53, ...}} 
- [{'Date': '...', 'Open': ..., ...}, ...]
- Dict date→price mapping
3. Retry mechanism không giải quyết đúng vấn đề
Retry chỉ đổi từ "4 decimals" → "2 decimals" mà không hề nói "KHÔNG output Python dict hay JSON". Cần một retry instruction mạnh hơn.
4. Lookback càng dài, fail càng nhiều
- lb=1: 1.06% fail
- lb=7: 2.87% fail
- lb=14: 9.14% fail
- lb=21: 1.65% fail
- lb=30: 38.26% fail ← cực cao
Với lb=30, input dài ~180+ tokens → attention của model 2.7B bị quá tải, mất instruction về format output.
5. Thiếu few-shot example cụ thể
Prompt chỉ có TEMPLATE/EXAMPLE trừu tượng. Không có ví dụ "input JSON → output đúng" thực tế.
6. num_predict=1500 có thể bị truncate
Output dài (nhất là khi model xuất dict 60+ entries) bị cắt ngang → không parse được.
Tóm lại: Model quá nhỏ + input dài + instruction không đủ mạnh + retry sai hướng = fail rate cao, đặc biệt ở lookback 30.

Còn gemma4:e2b thì ngay từ lookback 1 đã gặp parse fail khá cao rồi.
---
## Docs: update table compare parse fail rate & result between 3 models

Tổng kết Parse Fail & So sánh Model
### 1. Tỷ lệ Parse Failure

| Model | Tổng Fail | Tỷ lệ |
| --- | --- | --- |
| Gemma3 4B (run 3.5) | 0 / 2150 | 0.0% ✅ |
| Gemma3 4B (run 0) | 10 / 2170 (lookback 28) | 0.5% |
| Gemma4 E2B (run 1) | 103 / 1075 | 9.6% |
| Gemma3 4B (run 1) | 278 / 2150 | 12.9% |
| Gemma3 4B (run 2) | 287 / 2150 | 13.3% |
| Phi 2.7B (run 0) | 703 / 2064 | 34.1% ❌ (chưa config) |
| Phi 2.7B (run 1) | 298 / 2195 | 13.6% |

### 2. Chỉ số Metrics (trung bình các lookback)

| Model | Step 1 MAE | Step 30 MAE | Step 30 MAPE |
| --- | --- | --- | --- |
| Gemma3 4B (run 3.5) | 6.60 | 42.39 | 11.88% |
| Gemma3 4B (run 0) | 9.00 | 51.25 | 12.68% |
| Gemma4 E2B (run 1) | 8.03 | 44.81* | 11.51%* |
| Gemma3 4B (run 1) | 10.92 | 42.94 | 11.97% |
| Gemma3 4B (run 2) | 10.93 | 42.57 | 11.97% |
| Phi 2.7B (run 0) | 173.68 | 195.67 | 37.86% |
| Phi 2.7B (run 1) | 214.53 | CATASTROPHIC | CATASTROPHIC |

*Gemma4 E2B loại trừ lb30 vì có numerical explosion (MAE = 72 tỷ)

### 3. Pattern Parse Fail theo Lookback

| Model | lb1 | lb7 | lb14 | lb21 | lb30 |
| --- | --- | --- | --- | --- | --- |
| Gemma3 (run 3.5) | 0 | 0 | 0 | 0 | 0 |
| Gemma3 (run 0) | 0 | 0 | 0 | 2 | 8 |
| Gemma3 (run 1) | ~1% | ~8% | ~12% | ~18% | ~28% |
| Gemma4 E2B | ~5% | ~8% | ~10% | ~12% | 85% |
| Phi 2.7B (run 0) | 1% | 3% | 9% | 2% | 38% |
### 4. Nhận xét chính
- Winner: Gemma3 4B (run 3.5) — 0 parse fail, MAE thấp nhất, ổn định nhất
- Gemma3 càng cải thiện qua các run — run 1/2 fail ~280 lần, run 3.5 = 0 lần → do fix prompt
- Gemma4 E2B — lb30 catastrophic (98/115 fail), numerical explosion ra hàng tỷ
- Phi 2.7B — quá nhỏ, output Python dict thay vì số, MAE tệ gấp 30-50x Gemma3, lb30 fail 38%
- Xu hướng chung: lookback càng dài → parse fail càng cao (model bị quá tải context)

### update main_improved:
4 cải thiện chính:
|	| Cải thiện | Mô tả |
| --- | --- | --- |
|1 | Technical Indicators | Input có MA5/10/20, RSI, Volatility, ATR thay vì raw OHLCV → model dễ hiểu pattern
|2 | Few-shot Examples | Prompt có ví dụ thực tế input JSON → output đúng → model hiểu format mong muốn
|3 | Compact Input | JSON nén (key ngắn: c, o, h, l, v) → giảm token count → ít overload
|4 | Smart Retry | Retry message nhấn mạnh đúng lỗi ("KHÔNG output dict/JSON") thay vì chỉ đổi decimals