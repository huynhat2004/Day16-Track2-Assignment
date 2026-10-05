# Báo cáo ngắn - Lab 16 AWS Cloud AI Environment

1. Hạ tầng AWS được triển khai bằng Terraform gồm VPC, public/private subnet, Bastion Host, NAT Gateway, ALB và Compute Node.
2. Do tài khoản AWS giới hạn Free Tier, Compute Node được chạy với instance `t3.micro` thay vì `t3.medium`.
3. Dataset sử dụng là Credit Card Fraud Detection của Kaggle với 284,807 giao dịch và 31 cột.
4. Thời gian load dữ liệu đạt 2.50 giây, cho thấy dataset có thể đọc khá nhanh trên CPU node nhỏ.
5. Thời gian huấn luyện LightGBM đạt 8.28 giây với 300 estimators.
6. AUC-ROC đạt 0.8011, thể hiện model có khả năng phân biệt giao dịch gian lận và bình thường ở mức khá.
7. Recall đạt 0.9286 nhưng precision thấp do dataset mất cân bằng mạnh giữa lớp gian lận và không gian lận.
8. Inference latency cho 1 dòng là 1.43 ms, phù hợp với tác vụ dự đoán nhanh trên CPU.
9. Throughput với batch 1000 dòng đạt khoảng 75,039 rows/s, cho thấy LightGBM rất nhẹ và hiệu quả cho inference CPU.
10. Sau khi hoàn thành chụp ảnh và nộp bài, cần chạy `terraform destroy` để tránh phát sinh chi phí AWS.
