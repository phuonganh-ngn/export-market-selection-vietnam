# Tóm tắt kết quả mở rộng MLP + TabM

**Trạng thái: COMPLETE_REDUCED_SPECIFICATION.**

| Nhiệm vụ | Baseline tốt nhất | MLP | TabM | Kết quả quan sát | Kết luận thống kê |
|---|---|---:|---:|---|---|
| Phát hiện thị trường | LightGBM PR-AUC 0.3827 | 0.3451 | 0.3576 | LightGBM đứng đầu PR-AUC | MLP−LightGBM: Δ=-0.0376, CI95% [-0.0578, -0.0121], pHolm=0.024 (có khác biệt sau Holm); MLP−Gravity_PPML: Δ=-0.0322, CI95% [-0.0494, -0.0103], pHolm=0.006 (có khác biệt sau Holm); TabM−LightGBM: Δ=-0.0252, CI95% [-0.0464, 0.0021], pHolm=0.154 (chưa phân biệt được sau Holm); TabM−Gravity_PPML: Δ=-0.0197, CI95% [-0.0377, 0.0026], pHolm=0.154 (chưa phân biệt được sau Holm); TabM−MLP: Δ=0.0125, CI95% [0.0042, 0.0213], pHolm=0.024 (có khác biệt sau Holm) |
| Xếp hạng — thị trường mới | Gravity_PPML 0.4370 | 0.2801 | 0.2935 | Gravity_PPML đứng đầu | TabM−Gravity_PPML: Δ=-0.1435, CI95% [-0.1803, -0.1082], pHolm=<0.001 (có khác biệt sau Holm); TabM−LightGBM: Δ=-0.0094, CI95% [-0.0371, 0.0146], pHolm=0.394 (chưa phân biệt được sau Holm); MLP−Gravity_PPML: Δ=-0.1569, CI95% [-0.1908, -0.1217], pHolm=<0.001 (có khác biệt sau Holm); MLP−LightGBM: Δ=-0.0228, CI95% [-0.0475, -0.0012], pHolm=0.134 (chưa phân biệt được sau Holm); TabM−MLP: Δ=0.0134, CI95% [-0.0012, 0.0266], pHolm=0.137 (chưa phân biệt được sau Holm) |
| Xếp hạng — mở rộng hiện hữu | Gravity_PPML 0.5925 | 0.3654 | 0.3324 | Gravity_PPML đứng đầu | TabM−Gravity_PPML: Δ=-0.2601, CI95% [-0.3156, -0.2262], pHolm=<0.001 (có khác biệt sau Holm); TabM−LightGBM: Δ=-0.2081, CI95% [-0.2649, -0.1735], pHolm=<0.001 (có khác biệt sau Holm); MLP−Gravity_PPML: Δ=-0.2272, CI95% [-0.2720, -0.1933], pHolm=<0.001 (có khác biệt sau Holm); MLP−LightGBM: Δ=-0.1752, CI95% [-0.2232, -0.1387], pHolm=<0.001 (có khác biệt sau Holm); TabM−MLP: Δ=-0.0329, CI95% [-0.0599, -0.0176], pHolm=<0.001 (có khác biệt sau Holm) |
| Dự báo giá trị xuất khẩu (MAE USD) | Persistence 6,493,933 | 10,285,795 | 11,069,578 | Persistence đứng đầu MAE | MLP−Persistence: Δ=3,791,862, CI95% [1,635,646, 6,858,660], pHolm=<0.001 (có khác biệt sau Holm); TabM−Persistence: Δ=4,575,646, CI95% [1,752,760, 9,204,715], pHolm=<0.001 (có khác biệt sau Holm); MLP−Gravity_PPML: Δ=2,428,515, CI95% [1,092,225, 4,294,636], pHolm=0.548 (chưa phân biệt được sau Holm); TabM−Gravity_PPML: Δ=3,212,298, CI95% [1,131,472, 6,601,616], pHolm=0.308 (chưa phân biệt được sau Holm) |

Mẫu ex-ante gồm 62,797 quan sát; nhãn đã biết 62,238; thiếu 559. Kết quả chính là complete-case và chưa có độ nhạy neural một năm do tập ứng viên thay đổi.

Năm đánh giá chỉ có ba; WDI/WGI là vintage lịch sử đã được hiệu chỉnh. Không diễn giải xếp hạng quan sát như lợi ích xuất khẩu nhân quả hay khuyến nghị ở cấp doanh nghiệp.

Tệp chi tiết: `E1_FORECASTING_RESULTS.md`, `E2_DISCOVERY_RESULTS.md`, `E3_RANKING_RESULTS.md`, `STATISTICAL_INFERENCE.md`, `MISSING_LABEL_SENSITIVITY.md`.
