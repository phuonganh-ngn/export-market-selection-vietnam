# Tóm tắt điều hành

Phần mở rộng này kiểm tra lựa chọn thị trường dưới ba bất định tách biệt: trọng số mục tiêu chưa rõ, dự báo mô hình không đồng nhất, và một số nhãn đánh giá bị thiếu. Nhãn `PR-EMS` trong mã là minimax regret chuẩn trên một tập kịch bản hữu hạn, không phải thuật toán mới.

Mẫu đầu vào gồm 62,797 cặp nước đến–HS2–năm; Task A có 41,131, Task B có 12,962, giao nhau 6,511, thuộc cả hai không 15,215. Có 559 kết quả chưa quan sát và không bị gán bằng 0.

## Kết quả K=10 (utility chuẩn hóa; không phải USD)

| Nhiệm vụ | Chính sách | Utility trung bình | Regret tệ nhất trung bình | Đủ K nhãn quan sát | Đủ mọi slot khả dụng | Pool có <K ứng viên |
|---|---|---:|---:|---:|---:|---:|
| Task_A_new_entry | LightGBM | 1.7273 | 2.0322 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | XGBoost | 1.7255 | 2.0332 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | Standard_minimax_regret | 1.6807 | 2.0938 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | RandomForest | 1.6682 | 2.1000 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | Relative_robust_Weber | 1.6680 | 2.1089 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | Equal_weight_MCDM | 1.6641 | 2.1143 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | Uniform_scenario_expected_utility | 1.6641 | 2.1143 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | PR-EMS_linear_amount_utility | 1.6540 | 2.1263 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | PR-EMS | 1.6337 | 2.1503 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | MLP | 1.6296 | 2.1559 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | Gravity_PPML | 1.6137 | 2.1672 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | TabM | 1.6169 | 2.1708 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | LogisticRegression | 1.6050 | 2.1839 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | LambdaMART | 1.5630 | 2.2279 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | Maximin_preference | 1.5394 | 2.2634 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | Borda_rank_aggregation | 1.1933 | 2.6881 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | World_import_demand | 1.1973 | 2.6931 | 1.0000 | 1.0000 | 0.0000 |
| Task_A_new_entry | Joint_maximin_utility | 0.6082 | 3.3963 | 0.9375 | 0.9375 | 0.0000 |
| Task_A_new_entry | Predictive_maximin | 0.6127 | 3.3986 | 0.9583 | 0.9583 | 0.0000 |
| Task_A_new_entry | Historical_entry_frequency | 0.2941 | 3.7657 | 0.9028 | 0.9028 | 0.0000 |
| Task_B_existing_underexploited | Maximin_preference | 5.0464 | 2.8612 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | Relative_robust_Weber | 5.0398 | 2.8643 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | Equal_weight_MCDM | 5.0243 | 2.8796 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | Uniform_scenario_expected_utility | 5.0243 | 2.8796 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | Standard_minimax_regret | 5.0078 | 2.9010 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | PR-EMS | 4.9186 | 2.9948 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | PR-EMS_linear_amount_utility | 4.8806 | 3.0223 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | Borda_rank_aggregation | 4.8784 | 3.0390 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | LambdaMART | 4.8574 | 3.0625 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | Gravity_PPML | 4.8582 | 3.0840 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | Historical_import_demand | 4.8222 | 3.1046 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | LightGBM | 4.7818 | 3.1377 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | XGBoost | 4.7758 | 3.1459 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | MLP | 4.6830 | 3.2311 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | GDP | 4.6737 | 3.2467 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | TabM | 4.5452 | 3.4477 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | Joint_maximin_utility | 4.2455 | 3.7686 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | Predictive_maximin | 4.2240 | 3.7894 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | RandomForest | 3.9955 | 4.0317 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | Persistence | 3.5967 | 4.4350 | 0.9792 | 1.0000 | 0.0208 |
| Task_B_existing_underexploited | Historical_export_trend | 3.0211 | 5.0454 | 0.9792 | 1.0000 | 0.0208 |

Kết luận về chính sách phải dựa trên các bảng paired và khoảng bất định, không chỉ vào utility trung bình. Các lựa chọn trọng số là giả định để phân tích độ nhạy, không phải ưu tiên được hỏi từ doanh nghiệp.

Task B có sáu pool K=10 với ít hơn 10 ứng viên hợp lệ; do đó tỷ lệ đủ K nhãn quan sát được là 97,92%, trong khi mọi slot khả dụng đều có nhãn quan sát và Task B không có nhãn thiếu. Đây là giới hạn kích thước pool, không phải thiếu dữ liệu kết quả.

Không có tập holdout độc lập: 2022–2024 đã được xem trong quá trình nghiên cứu trước và BACI HS12 V202601 kết thúc ở 2024. Do đó kết quả chỉ mang tính khám phá. Cần có năm dữ liệu mới chưa từng dùng hoặc dữ liệu quyết định ở bối cảnh khác trước khi khẳng định chính sách robust cải thiện hiệu quả.
