# Kết quả các model

Tất cả đánh giá bằng cross-validation 5 fold (stratified, `random_state=42`) trên `train.csv` (266 row).
Tập test chưa được dùng: chỉ đánh giá một lần ở cuối cho model được chọn.

| Model | Accuracy | Macro-F1 | Script |
| :--- | ---: | ---: | :--- |
| Baseline (`DummyClassifier(strategy="most_frequent")`) | 0.44 (± 0.01) | 0.20 (± 0.00) | `src/baseline.py` |
| KNN raw (k=5, 4 số đo) | 0.76 (± 0.08) | 0.71 (± 0.09) | `src/experiment_scaling.py` |
| KNN + scaler (k=5, 4 số đo) | 0.98 (± 0.01) | 0.98 (± 0.01) | `src/experiment_scaling.py` |
| KNN + scaler, tuned (k=5, uniform, euclidean) | 0.98 (± 0.01) | 0.98 (± 0.01) | `src/tune.py` |
| KNN tuned + sex + island (one-hot) | 0.99 (± 0.01) | 0.99 (± 0.01) | `src/experiment_features.py` |
| **Candidate:** KNN + sex + island, tuned lại (k=5, distance, manhattan) | 1.00 (± 0.00) | 1.00 (± 0.00) | `src/select_model.py` |

## Baseline

Model luôn đoán **Adelie**, loài nhiều nhất trong train (117/266 con).

- **Accuracy 0.44:** đúng bằng tỉ lệ Adelie, vì model chỉ đúng khi gặp Adelie.
- **Macro-F1 0.20:** F1 của Adelie ≈ 0.61, còn Chinstrap và Gentoo đều bằng 0 vì không bao giờ được đoán. Trung bình 3 loài: 0.61 / 3 ≈ 0.20.

Accuracy 0.44 nghe có vẻ không quá tệ, nhưng Macro-F1 0.20 cho thấy model hoàn toàn vô dụng với 2/3 số loài. Vì vậy Macro-F1 là chỉ số chính để so sánh.

**Mốc cần vượt:** các model sau (KNN, ...) phải đạt Accuracy trên 0.44 và Macro-F1 trên 0.20, đánh giá cùng cách chia fold như trên.

## Thí nghiệm 1: không scale và có scale

Hai version dùng cùng KNN mặc định (k=5), cùng 4 số đo, cùng cách chia fold; chỉ khác nhau ở bước `StandardScaler`.
Kết quả chi tiết: `reports/experiment_scaling.csv`.

| Model | Accuracy | Macro-F1 |
| :--- | ---: | ---: |
| KNN raw | 0.76 (± 0.08) | 0.71 (± 0.09) |
| KNN + scaler | 0.98 (± 0.01) | 0.98 (± 0.01) |

**Kết luận:** scale tăng Macro-F1 từ 0.71 lên 0.98, đồng thời kết quả ổn định hơn nhiều giữa các fold (độ lệch chuẩn từ 0.09 xuống 0.01). Từ đây mọi model KNN đều dùng `StandardScaler` trong `Pipeline`.

**Vì sao KNN raw kém:** KNN chọn láng giềng theo khoảng cách Euclid, nên cột có khoảng giá trị lớn sẽ lấn át các cột khác. Độ lệch chuẩn của các cột trong train:

| Cột | Độ lệch chuẩn |
| :--- | ---: |
| `bill_length_mm` | 5.5 |
| `bill_depth_mm` | 2.0 |
| `flipper_length_mm` | 14.0 |
| `body_mass_g` | 810.4 |

`body_mass_g` lớn hơn các cột khác 60 đến 400 lần, nên khoảng cách gần như chỉ phụ thuộc vào khối lượng. Theo EDA, khối lượng chỉ tách được Gentoo, còn Adelie và Chinstrap nặng gần bằng nhau (khoảng 3.700 g). Vì vậy KNN raw nhầm nhiều nhất ở Chinstrap: recall chỉ 0.39 (21/54 con đoán đúng, 31 con bị đoán thành Adelie), trong khi Gentoo vẫn đạt recall 0.92.

Kiểm chứng thêm: KNN raw chỉ dùng mỗi `body_mass_g` đạt Macro-F1 0.59, khá gần mức 0.71 của KNN raw dùng cả 4 cột. Nghĩa là khi chưa scale, ba số đo còn lại chỉ đóng góp một phần nhỏ.

Sau khi scale, mỗi cột có độ lệch chuẩn bằng 1, nên số đo mỏ (thứ tách được Adelie và Chinstrap) mới được KNN tính đến.

## Thí nghiệm 2: chọn config bằng GridSearchCV

Thử 28 config (`StandardScaler` + KNN), chọn theo Macro-F1 trung bình 5 fold trên train:

- `n_neighbors` (K): 3, 5, 7, 9, 11, 13, 15
- `weights`: uniform, distance
- `metric`: euclidean, manhattan

Kết quả chi tiết: `reports/tuning_results.csv`. Config tốt nhất lưu ở `reports/best_params.json`.

**Best params:** K = 5, weights = uniform, metric = euclidean
Accuracy 0.9850 (± 0.0075), Macro-F1 0.9814 (± 0.0093).

Đây trùng với config mặc định của `KNeighborsClassifier`, nên điểm số giống hệt "KNN + scaler" ở thí nghiệm 1.

![Macro-F1 CV theo K](figures/k_vs_cv_score.png)

Đường là Macro-F1 trung bình 5 fold; dải mờ là ± độ lệch chuẩn giữa các fold. Các dải chồng lên nhau gần như hoàn toàn, tức là chênh lệch giữa các config nhỏ hơn độ dao động giữa các fold.

Macro-F1 theo từng tổ hợp weights/metric:

| K | uniform + euclidean | distance + euclidean | uniform + manhattan | distance + manhattan |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.9765 | 0.9722 | 0.9675 | 0.9675 |
| 5 | **0.9814** | 0.9770 | 0.9722 | 0.9678 |
| 7 | 0.9714 | 0.9671 | 0.9714 | 0.9671 |
| 9 | 0.9714 | 0.9671 | 0.9714 | 0.9671 |
| 11 | 0.9720 | 0.9671 | 0.9763 | 0.9671 |
| 13 | 0.9720 | 0.9671 | 0.9720 | 0.9671 |
| 15 | 0.9720 | 0.9671 | 0.9720 | 0.9671 |

**Nhận xét:**

- **Mọi config đều rất tốt và gần nhau:** Macro-F1 chỉ dao động từ 0.967 đến 0.981. Tính theo accuracy, config tốt nhất (0.9850) và kém nhất (0.9737) chỉ chênh 3 con trên 266 con đoán đúng.
- **Chênh lệch nhỏ hơn độ dao động giữa các fold:** độ lệch chuẩn của nhiều config là 0.01 đến 0.03, lớn hơn khoảng cách giữa config tốt nhất và các config khác. Vì vậy không thể khẳng định K=5 thực sự tốt hơn, ví dụ, K=3 hoặc K=11. Dữ liệu này chỉ đủ để kết luận: sau khi scale, KNN hoạt động tốt với hầu hết mọi config hợp lý.
- **K:** K=3 và K=5 nhỉnh hơn một chút. Từ K=7 trở lên điểm gần như không đổi, do các loài tách thành cụm rõ ràng nên thêm láng giềng cũng không đổi đáp án.
- **weights:** uniform bằng hoặc nhỉnh hơn distance ở mọi K. Với K lớn, distance cho kết quả giống hệt nhau ở cả hai metric.
- **metric:** euclidean nhỉnh hơn manhattan ở K=3 và K=5, còn lại gần như bằng nhau.

**Quyết định:** dùng K = 5, weights = uniform, metric = euclidean. Config này vừa có Macro-F1 cao nhất vừa ổn định nhất giữa các fold (độ lệch chuẩn thấp nhất: 0.0093), và cũng là config đơn giản nhất (mặc định của sklearn).

## Thí nghiệm 3: thêm sex và island

Hai version dùng cùng config KNN tốt nhất (K=5, uniform, euclidean, lấy từ `reports/best_params.json`) và cùng cách chia fold; chỉ khác nhau ở tập feature.
Tiền xử lý bằng `ColumnTransformer` nằm trong `Pipeline`, nên chỉ học từ phần train của từng fold:

- 4 cột số → `StandardScaler`
- `sex`, `island` → `OneHotEncoder(handle_unknown="ignore")`

Kết quả chi tiết: `reports/experiment_features.csv`.

| Feature set | Accuracy | Macro-F1 |
| :--- | ---: | ---: |
| Numeric only | 0.9850 (± 0.0075) | 0.9814 (± 0.0093) |
| Numeric + sex + island | 0.9925 (± 0.0092) | 0.9911 (± 0.0109) |

**Số con đoán sai** (gộp 5 fold, trên 266 con):

| Feature set | Số con sai | Nhầm như thế nào |
| :--- | ---: | :--- |
| Numeric only | 4 | 4 Chinstrap bị đoán thành Adelie |
| Numeric + sex + island | 2 | 1 Chinstrap thành Adelie, 1 Adelie thành Chinstrap |

**Nhận xét:**

- Thêm `sex` và `island` giảm số con đoán sai từ 4 xuống 2, Macro-F1 tăng từ 0.9814 lên 0.9911. Gentoo vẫn đúng 100% ở cả hai version.
- Mức tăng này chỉ tương ứng 2 con, nhỏ hơn độ dao động giữa các fold (std khoảng 0.01), nên chỉ có thể nói là xu hướng tốt hơn, chưa thể khẳng định chắc chắn.
- Cả 4 con Chinstrap bị đoán sai ở version numeric đều là **con cái**, có mỏ ngắn (40.9 đến 46.0 mm) và nhẹ. Chúng rơi vào vùng số đo của Adelie, đúng vùng chồng lấn đã thấy ở biểu đồ EDA số 6.

**Feature nào thực sự giúp?** Tách riêng từng feature (thí nghiệm phụ, cùng cấu hình):

| Feature set | Macro-F1 | Số con sai |
| :--- | ---: | ---: |
| Numeric only | 0.9814 | 4 |
| Numeric + island | 0.9819 | 4 |
| Numeric + sex | 0.9909 | 2 |
| Numeric + sex + island | 0.9911 | 2 |

- **`sex` là feature tạo ra cải thiện.** Con cái nhỏ hơn con đực ở mọi loài. Khi không biết giới tính, một con Chinstrap cái nhỏ trông giống Adelie. Khi biết giới tính, KNN ưu tiên so với các con cùng giới, nên phân biệt tốt hơn.
- **`island` gần như không giúp thêm.** Dù EDA cho thấy `island` tách loài rất mạnh (Chinstrap chỉ ở Dream), mọi ca khó đều là Adelie và Chinstrap trên đảo Dream, nơi cả hai loài cùng sống. Vì vậy `island` không phân biệt được đúng những con mà model đang nhầm.

**Lưu ý về thang đo:** sau one-hot, mỗi cột phân loại có giá trị 0 hoặc 1 và không được scale. Hai con khác đảo sẽ cách nhau thêm √2 ≈ 1.41 đơn vị trong khoảng cách Euclid, tương đương khoảng 1.4 độ lệch chuẩn của một số đo. Vì vậy trọng số của `sex` và `island` khá lớn so với các số đo, và KNN gần như ưu tiên tìm láng giềng cùng giới, cùng đảo.

## Chốt model candidate

Config K=5 ở thí nghiệm 3 được tune trên bộ 4 số đo. Để so sánh công bằng, `src/select_model.py` chạy lại cùng lưới 28 config (K, weights, metric) cho **từng** bộ feature, rồi chọn theo Macro-F1 CV. Nếu bằng điểm thì ưu tiên std thấp hơn, sau đó ưu tiên bộ feature đơn giản hơn.

Kết quả chi tiết: `reports/model_selection.csv`.

| Feature set | K | metric | weights | Accuracy | Macro-F1 |
| :--- | ---: | :--- | :--- | ---: | ---: |
| numeric | 5 | euclidean | uniform | 0.9850 (± 0.0075) | 0.9814 (± 0.0093) |
| numeric + sex + island | 5 | manhattan | distance | **1.0000 (± 0.0000)** | **1.0000 (± 0.0000)** |

**Model candidate** (lưu ở `ml/config.json`):

- Features: 4 số đo + `sex` + `island`
- K = 5
- metric = manhattan
- weights = distance

Model này đoán đúng cả 266 con trong CV, trên cả 5 fold.

**Manhattan có thực sự tốt hơn Euclid không?** Chưa thể khẳng định. Với cùng K=5 và weights=distance, euclidean đạt Macro-F1 0.9956, manhattan đạt 1.0000, chênh nhau đúng 1 con trên 266. Mức chênh này nhỏ hơn độ dao động giữa các fold, nên việc manhattan đứng đầu chủ yếu là do may mắn với cách chia fold này, không phải vì manhattan phù hợp hơn hẳn với dữ liệu.

**Cần thận trọng với điểm 1.0:**

- Điểm này là điểm **tốt nhất trong 28 config**, được chọn dựa trên chính các fold dùng để chấm. Khi chọn config tốt nhất trong nhiều lựa chọn, điểm của nó thường cao hơn thực tế một chút (lạc quan hơn so với dữ liệu mới).
- Khoảng cách tới version numeric chỉ là 4 con trên 266.
- Vì vậy điểm trên tập test nhiều khả năng sẽ thấp hơn 1.0 một chút. Tập test (67 con, chưa dùng lần nào) sẽ cho con số khách quan.

Kiểm tra thêm sau khi chốt: với bộ numeric + sex + island, chỉ có 2/28 config đạt Macro-F1 CV 1.0 (K=5 và K=7, cùng manhattan + distance). 26 config còn lại đạt 0.9909 đến 0.9956, tức chỉ sai 1 đến 2 con. GridSearch chọn K=5 vì nó đứng trước K=7 trong lưới. Việc chọn config trong bộ feature này gần như không ảnh hưởng tới kết quả.

## Đánh giá cuối trên test

Model candidate (`ml/config.json`) được huấn luyện trên toàn bộ train (266 con), rồi chấm **một lần** trên test (67 con, chưa dùng ở bất kỳ bước nào trước đó). Script: `src/evaluate.py`.
Kết quả lưu ở `reports/metrics.json` và `reports/confusion_matrix.png`.

| Tập | Accuracy | Macro-F1 |
| :--- | ---: | ---: |
| CV trên train (lúc chọn model) | 1.0000 | 1.0000 |
| **Test** | **0.9851** | **0.9827** |

![Confusion matrix trên test](confusion_matrix.png)

| Loài | Precision | Recall | F1 | Số con |
| :--- | ---: | ---: | ---: | ---: |
| Adelie | 1.0000 | 0.9655 | 0.9825 | 29 |
| Chinstrap | 0.9333 | 1.0000 | 0.9655 | 14 |
| Gentoo | 1.0000 | 1.0000 | 1.0000 | 24 |

**Nhận xét:**

- Model đoán đúng 66/67 con. Gentoo đúng 100%.
- **Con duy nhất bị sai:** một con Adelie cái trên đảo Dream (mỏ dài 42.2 mm, sâu 18.5 mm, nặng 3.550 g) bị đoán thành Chinstrap. Mỏ dài 42.2 mm là dài so với Adelie, rơi vào vùng chồng lấn Adelie–Chinstrap đã thấy ở EDA. Đây cũng đúng là kiểu nhầm lẫn duy nhất đã gặp ở các thí nghiệm CV trước (Adelie ↔ Chinstrap cái trên đảo Dream).
- **Điểm test thấp hơn CV** (0.9827 so với 1.0), đúng như dự đoán ở bước chốt model: điểm CV 1.0 được chọn là tốt nhất trong 28 config nên hơi lạc quan. Mức chênh chỉ là 1 con. Test chỉ có 67 con, nên mỗi con sai làm accuracy giảm khoảng 0.015.
- So với baseline (Macro-F1 0.20) và KNN raw (0.71), model cuối tốt hơn rất rõ.

Sau bước này, model không được chỉnh sửa gì dựa trên kết quả test.

## Thí nghiệm 4: error analysis

Script `src/error_analysis.py` lấy các con bị đoán sai trên test, rồi tìm 5 hàng xóm gần nhất trong train. Hàng xóm được tìm trong đúng không gian mà KNN dùng để dự đoán: số đo đã scale, `sex` và `island` đã one-hot, khoảng cách manhattan.
Kết quả chi tiết: `reports/error_analysis.csv` (dòng `sample` là con bị sai, các dòng `neighbor` là hàng xóm).

Chỉ có 1 con bị sai trên test:

**Actual: Adelie → Predicted: Chinstrap**
Adelie cái, đảo Dream, mỏ dài 42.2 mm, mỏ sâu 18.5 mm, chân chèo 180 mm, nặng 3.550 g.

| # | Distance | Loài | Mỏ dài | Mỏ sâu | Chân chèo | Khối lượng | Giới | Đảo |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| 1 | 0.7787 | Chinstrap | 42.4 | 17.3 | 181 | 3600 | female | Dream |
| 2 | 1.3784 | Chinstrap | 46.1 | 18.2 | 178 | 3250 | female | Dream |
| 3 | 1.4105 | Chinstrap | 42.5 | 17.3 | 187 | 3350 | female | Dream |
| 4 | 1.4521 | Adelie | 36.6 | 18.4 | 184 | 3475 | female | Dream |
| 5 | 1.5305 | Adelie | 38.9 | 18.8 | 190 | 3600 | female | Dream |

- **Số phiếu:** 3/5 hàng xóm là Chinstrap. Với `weights="distance"`, mỗi hàng xóm bỏ phiếu với trọng số 1/distance. Chinstrap được 67% phiếu, Adelie 33%, nên model đoán Chinstrap.
- **Vì sao sai:** con này có **mỏ dài bất thường so với Adelie**. Trong train có 20 con Adelie cái ở đảo Dream, mỏ dài nhất chỉ 40.2 mm (trung bình 37.2 mm). Con này mỏ dài 42.2 mm, gần với Chinstrap cái hơn (ngắn nhất 40.9 mm, trung bình 46.5 mm).
- **Chiều dài mỏ quyết định khoảng cách:** sau khi scale, mỗi đơn vị khoảng cách tương đương 1 độ lệch chuẩn (mỏ dài khoảng 5.5 mm, mỏ sâu khoảng 2.0 mm). Với hàng xóm #1, mỏ dài chỉ chênh 0.2 mm (khoảng 0.04 đơn vị), chủ yếu chênh ở mỏ sâu (1.2 mm, khoảng 0.6 đơn vị), tổng 0.78. Với hàng xóm Adelie gần nhất (#4), mỏ dài chênh 5.6 mm (khoảng 1.0 đơn vị), nên xa hơn (1.45) dù mỏ sâu gần như bằng nhau.
- **Mỏ sâu lẽ ra giúp nhận ra Adelie**, nhưng không đủ. Mỏ sâu 18.5 mm của con này hợp với Adelie hơn, nhưng Adelie cái và Chinstrap cái ở Dream có mỏ sâu gần như giống hệt nhau (trung bình 17.8 và 17.7 mm), nên feature này không tách được hai nhóm.
- **`sex` và `island` không giúp gì ở ca này:** cả 5 hàng xóm đều là con cái ở đảo Dream, giống sample, nên hai feature này không tạo ra khác biệt.
- **Liên hệ với thí nghiệm 3:** hàng xóm #1 và #3 (index 85 và 123 trong train) chính là 2 trong 4 con Chinstrap cái mỏ ngắn từng bị đoán thành Adelie trong CV ở version chỉ dùng số đo. Đây là vùng chồng lấn thật giữa hai loài: Chinstrap cái mỏ ngắn và Adelie cái mỏ dài có số đo gần như trùng nhau.

**Kết luận:** lỗi này không phải do model hay tiền xử lý có vấn đề. Con chim này thực sự nằm trong vùng mà 4 số đo cùng `sex` và `island` không đủ để phân biệt Adelie và Chinstrap. Muốn sửa cần thêm thông tin khác (ví dụ hình dạng mỏ hoặc màu lông), không phải chỉnh K hay metric. Theo nguyên tắc đã đặt ra, model không được chỉnh sửa dựa trên phân tích này.
