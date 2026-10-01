# Data Dictionary – Palmer Penguins

Tài liệu này mô tả cấu trúc của bộ dữ liệu `data/raw/penguins.csv`.
Mỗi dòng dữ liệu đại diện cho một cá thể chim cánh cụt.

- Số dòng: 344
- Số cột: 8 (1 biến mục tiêu, 7 đặc trưng)
- Giá trị thiếu được ghi là `NA` trong file CSV.
- Kết quả kiểm tra chất lượng và cách xử lý giá trị thiếu: xem `data_quality.md`.

## `species`

- Kiểu dữ liệu: phân loại
- Vai trò: biến mục tiêu
- Đơn vị: không có
- Ý nghĩa: loài chim cánh cụt
- Giá trị: `Adelie`, `Gentoo`, `Chinstrap`
- Cho phép thiếu: không

## `island`

- Kiểu dữ liệu: phân loại
- Vai trò: đặc trưng
- Đơn vị: không có
- Ý nghĩa: hòn đảo nơi cá thể chim cánh cụt được quan sát
- Giá trị: `Torgersen`, `Biscoe`, `Dream`
- Cho phép thiếu: không

## `bill_length_mm`

- Kiểu dữ liệu: số thực
- Vai trò: đặc trưng
- Đơn vị: mm
- Ý nghĩa: chiều dài mỏ của chim cánh cụt
- Khoảng giá trị quan sát: 32,1–59,6 mm
- Cho phép thiếu: có (có 2 giá trị thiếu trong dữ liệu gốc)

## `bill_depth_mm`

- Kiểu dữ liệu: số thực
- Vai trò: đặc trưng
- Đơn vị: mm
- Ý nghĩa: độ sâu của mỏ chim cánh cụt
- Khoảng giá trị quan sát: 13,1–21,5 mm
- Cho phép thiếu: có (có 2 giá trị thiếu trong dữ liệu gốc)

## `flipper_length_mm`

- Kiểu dữ liệu: số nguyên (pandas đọc thành `float64` vì cột có `NA`)
- Vai trò: đặc trưng
- Đơn vị: mm
- Ý nghĩa: chiều dài chân chèo của chim cánh cụt
- Khoảng giá trị quan sát: 172–231 mm
- Cho phép thiếu: có (có 2 giá trị thiếu trong dữ liệu gốc)

## `body_mass_g`

- Kiểu dữ liệu: số nguyên (pandas đọc thành `float64` vì cột có `NA`)
- Vai trò: đặc trưng
- Đơn vị: g
- Ý nghĩa: khối lượng cơ thể của chim cánh cụt
- Khoảng giá trị quan sát: 2.700–6.300 g
- Cho phép thiếu: có (có 2 giá trị thiếu trong dữ liệu gốc)

## `sex`

- Kiểu dữ liệu: phân loại
- Vai trò: đặc trưng
- Đơn vị: không có
- Ý nghĩa: giới tính của chim cánh cụt
- Giá trị: `male`, `female`
- Cho phép thiếu: có (có 11 giá trị thiếu trong dữ liệu gốc)

## `year`

- Kiểu dữ liệu: số nguyên
- Vai trò: đặc trưng
- Đơn vị: năm
- Ý nghĩa: năm cá thể chim cánh cụt được quan sát
- Khoảng giá trị quan sát: 2007–2009
- Cho phép thiếu: không
