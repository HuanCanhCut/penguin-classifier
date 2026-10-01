# Kiểm tra chất lượng dữ liệu – Palmer Penguins

Kết quả kiểm tra `data/raw/penguins.csv` bằng script `src/check_data.py`.

## Tổng quan

- Số row: 344
- Số column: 8
- Số loài:
  - Adelie: 152
  - Gentoo: 124
  - Chinstrap: 68

## Giá trị thiếu

| Cột | Số giá trị thiếu |
| :--- | ---: |
| `sex` | 11 |
| `bill_length_mm` | 2 |
| `bill_depth_mm` | 2 |
| `flipper_length_mm` | 2 |
| `body_mass_g` | 2 |

Tổng cộng có 11 row bị thiếu dữ liệu, chia thành 2 nhóm:

1. **2 row thiếu toàn bộ số đo**: 1 Adelie (Torgersen, 2007) và 1 Gentoo (Biscoe, 2009).
   Hai row này chỉ còn `species`, `island`, `year`; cả 4 số đo và `sex` đều là `NA`.
2. **9 row chỉ thiếu `sex`**: 5 Adelie, 4 Gentoo. Các số đo vẫn đầy đủ.

## Trùng lặp

Không có row nào trùng lặp hoàn toàn (0 duplicate).

## Quyết định

### Bỏ 2 row thiếu toàn bộ số đo

- Mô hình dự đoán loài dựa vào số đo mỏ, chân chèo, khối lượng. Hai row này không còn số đo nào, nên không giúp gì cho việc học.
- Nếu điền giá trị (ví dụ lấy trung bình theo loài) thì đó là dữ liệu bịa hoàn toàn, lại còn dùng chính nhãn `species` để điền → rò rỉ nhãn (data leakage).
- Chỉ mất 2/344 row (0,6%).

### Bỏ 9 row thiếu `sex`

- Chỉ mất thêm 9 row (2,6%). Không loài nào mất nhiều: Chinstrap không mất row nào, còn Adelie và Gentoo mỗi loài mất khoảng 4%.
- Điền `sex` bằng giá trị phổ biến nhất sẽ đoán sai khoảng một nửa số trường hợp, vì tỉ lệ đực/cái gần bằng nhau.
- Cách làm đơn giản, dễ giải thích; dữ liệu còn lại vẫn đủ lớn cho bài toán 3 lớp.

Phương án khác: giữ các row này và coi `sex` thiếu là một nhóm riêng (`unknown`). Chưa chọn cách này vì nó thêm một nhóm giả vào mô hình, trong khi lợi ích chỉ là giữ thêm 9 row.

### Trùng lặp

Không có row trùng nên không cần xử lý. Script vẫn kiểm tra mỗi lần chạy để phát hiện nếu dữ liệu thay đổi.

## Kết quả sau khi làm sạch

Chạy `python ml/src/clean_data.py` để tạo `data/processed/penguins_clean.csv`.
File gốc trong `data/raw/` không bị sửa.

- Số row: 333 (bỏ 11 row)
- Adelie: 146
- Gentoo: 119
- Chinstrap: 68

Các lớp vẫn lệch nhau (Chinstrap ít nhất), nên khi chia train/test cần dùng `stratify` theo `species`.
