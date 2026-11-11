# Tóm tắt điều hành

Trên 288 pool HS2–năm mỗi tác vụ, NDCG@10 Task A là LightGBM 0.2947, PPML 0.2801, và phối hợp hạng đều 0.3319. Blend hơn LightGBM +0,0372 (KTC HS2-block 95%: 0,0058 đến 0,0710; p Holm=0,0272), nhưng Precision@10 và số entry không khác biệt có ý nghĩa. Chênh lệch cơ hội USD quan sát được so với LightGBM là $11,448/pool (KTC 95%: $-89,534 đến $118,882; p Holm=0.708; trung vị 0). Ở Task B, PPML (0.5925) cao hơn blend (0.5713) về NDCG và cơ hội USD.

Task A dùng ba năm không xuất khẩu đã xác minh đến mốc y−2; 556 nhãn tương lai vẫn để thiếu. Ở ngưỡng danh nghĩa 80%, coverage thực tế trung bình là 95,5% cho cổng risk blend Task A và 74,3% cho cổng PPML Task B. Cổng Task B hoãn 74 pool có tổng cơ hội Top-10 quan sát khoảng USD 4,29 tỷ; đây là cơ hội chưa được quyết định, không phải xuất khẩu bị mất. Không có dữ liệu chuyên gia, chi phí rà soát hay holdout độc lập; chưa thể khẳng định lợi ích triển khai hoặc hiệu quả hợp tác người–AI.

Phân loại bằng chứng: EXPLORATORY_ONLY; không có holdout độc lập mới. Kết quả hỗ trợ một bài nghiên cứu thực nghiệm thận trọng về tính bổ sung thứ hạng và giới hạn của cổng rủi ro, chưa hỗ trợ tuyên bố thuật toán mới hoặc lợi ích kinh tế nhân quả.
