# Tóm tắt điều hành ERGM

## Hệ thống làm gì

ERGM chọn giữa Gravity–PPML và LightGBM cho từng nhóm sản phẩm HS2 × năm. Sau đó hệ thống xếp hạng toàn bộ danh sách thị trường Top-10 và có thể chuyển cả nhóm sang trạng thái cần chuyên gia xem xét nếu điểm rủi ro dự báo cao.

## Dữ liệu và sửa định nghĩa

Phân tích dùng dữ liệu BACI, panel HS2 và dự báo đã lưu của M1. Trong định nghĩa cũ, 6.511 khóa cùng lúc thỏa điều kiện ‘chưa đạt ngưỡng xuất khẩu USD 50.000’ và ‘đã có xuất khẩu dương’. Mở rộng lần này định nghĩa thị trường mới là không có xuất khẩu được xác minh trong cả ba năm trước; vì vậy Task A và Task B không còn giao nhau. Có 29.737 quan sát Task A và 12.962 quan sát Task B; 556 nhãn Task A vẫn không quan sát được và không được thay bằng 0.

## Kết quả thực nghiệm

| task                           | static_expert   |   static_ndcg10 |   static_regret10 |   adaptive_ndcg10 |   adaptive_regret10 |   adaptive_lgb_share |   blend_ndcg10 |   ergm_matched80_regret |   ergm_matched80_ndcg |   ergm_operational80_coverage |   ergm_operational80_regret |
|:-------------------------------|:----------------|----------------:|------------------:|------------------:|--------------------:|---------------------:|---------------:|------------------------:|----------------------:|------------------------------:|----------------------------:|
| Task_A_new_entry               | LightGBM        |        0.294694 |          0.458333 |          0.281714 |            0.458333 |             0.399306 |       0.331878 |                0.246753 |              0.300409 |                      0.951389 |                    0.404118 |
| Task_B_existing_underexploited | Gravity_PPML    |        0.592524 |          2.15278  |          0.579031 |            2.25     |             0.295139 |       0.571279 |                1.09957  |              0.592273 |                      0.722222 |                    0.866752 |

Các bảng coverage–risk và so sánh cùng mức coverage 80% nằm trong `results/T5_coverage_risk.csv` và `results/T5_matched_coverage_comparison.csv`. Giá trị USD là cơ hội thương mại quan sát được trong dữ liệu lịch sử, không phải lợi nhuận hay tác động do khuyến nghị.

## Giới hạn

Không có quyết định hoặc kết quả sau thẩm định của chuyên gia con người. Vì thế đây là tự động hóa có chọn lọc và phân luồng hồ sơ xem xét, chưa phải nghiên cứu hợp tác người–AI đã được xác nhận. Các năm 2022–2024 đã được dùng trong phát triển M1; kết quả chỉ mang tính hồi cứu thăm dò, không có holdout độc lập. Không có thuật toán mới hoặc bảo đảm xác suất an toàn được tuyên bố.
