# Tóm tắt điều hành

Mô hình hai phần tách xác suất đạt ngưỡng xuất khẩu 50.000 USD khỏi giá trị xuất khẩu khi đã đạt ngưỡng. Trên dữ liệu Việt Nam 2022–2024, mô hình này không vượt LightGBM hoặc rank blend: NDCG@10 lần lượt 0,2774, 0,2947 và 0,3319; qualified export USD@10 trung bình khoảng 443 nghìn, 535 nghìn và 546 nghìn USD mỗi HS2-năm. Đây là đánh giá hồi cứu thăm dò.

Thử nghiệm THA và IDN dùng BACI V202601 có thật; cả hai có 96 chương HS2 trong từng năm. Mô hình Việt Nam chuyển sang nước khác và mô hình fit riêng đều chạy được. Kết quả khác nhau theo nước, không chứng minh Two-Part tốt hơn; PPML vẫn mạnh ở nhiệm vụ mở rộng thị trường hiện hữu. THA đã lộ một số nhãn trong bước gỡ lỗi nên chỉ xem là thăm dò. IDN là đánh giá địa lý được định trước, không phải holdout theo thời gian độc lập.

Kết luận phù hợp nhất: hurdle/two-part chưa cho thấy lợi ích kinh tế tổng quát; việc đo riêng xác suất vào thị trường, quy mô có điều kiện và USD thực tế là cần thiết. Không có bằng chứng nhân quả, lợi nhuận doanh nghiệp hay phương pháp thuật toán mới.


Kiểm định ghép cặp theo cụm HS2 cho thấy blend cao hơn LightGBM 0,0372 điểm NDCG@10 (CI 95%: 0,0058–0,0710; p Holm=0,027), nhưng chưa tăng rõ Precision@10, số lượt vào thị trường hay USD đủ ngưỡng. Chênh qualified USD Top-10 blend–LightGBM là khoảng +11,274 USD mỗi pool (CI 95%: −89,488 đến +118,503; p Holm=1,00). Mô hình Two-Part thấp hơn blend về NDCG@10; chênh lệch −0,0545 (p Holm=0,0237) và không cho thấy tăng USD.

Trong mở rộng thị trường hiện hữu, PPML vẫn có NDCG@10 cao hơn các mô hình so sánh ở cả THA và IDN. Bảng luồng mẫu ngoài nước đã được sửa để không đếm hai lần ứng viên B1/B2. Phép tái dựng độc lập kiểm tra 16.144 ô kết quả; cả 26 kiểm thử đều đạt.
