# Eigenvalue Algorithms (Python)

Triển khai các thuật toán tìm trị riêng và vector riêng của ma trận bằng Python (NumPy).

## Danh sách thuật toán
- Power Iteration
- Inverse Iteration (No shift)
- Shifted Inverse Iteration
- Rayleigh Quotient Iteration (RQI)
- Simultaneous Iteration
- QR Algorithm

## Mô tả file
- `power_iteration.py`: Tìm trị riêng lớn nhất
- `inverse_iteration_noshift.py`: Tìm trị riêng nhỏ nhất
- `inverse_iteration_shift.py`: Tìm trị riêng gần shift μ
- `rayleigh_quotient_iteration.py`: Tìm một trị riêng bất kỳ (hội tụ bậc 3)
- `simultaneous_iteration.py`: Tìm p trị riêng lớn nhất
- `qr_method.py`: Tìm toàn bộ phổ trị riêng

## Yêu cầu
- Python 3.11
- NumPy

## Chạy ví dụ
```bash
python power_iteration.py
