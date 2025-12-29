import numpy as np

def qr_algorithm(A, tol=1e-10, max_iter=1000):
    n = A.shape[0]
    A_k = A.copy()
    V = np.eye(n)   # tích lũy vector riêng

    for k in range(max_iter):
        # 1. Phân tích QR
        Q, R = np.linalg.qr(A_k)

        # 2. Nhân đảo
        A_k = R @ Q

        # 3. Tích lũy vector riêng
        V = V @ Q

        # 4. Kiểm tra hội tụ (ngoài đường chéo)
        off_diag_norm = np.linalg.norm(
            A_k - np.diag(np.diag(A_k)), ord='fro'
        )

        if off_diag_norm < tol:
            print(f"Hội tụ tại bước lặp {k + 1}")
            break
    else:
        print("Không hội tụ trong số bước lặp cho phép.")

    # 5. Trích xuất kết quả
    eigenvalues = np.diag(A_k)
    eigenvectors = V

    # 6. Sắp xếp theo độ lớn trị riêng giảm dần
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


# Ví dụ sử dụng
if __name__ == "__main__":
    A = np.array([
        [4, 1, 1],
        [1, 3, 0],
        [1, 0, 2]
    ], dtype=float)

    eigenvalues, eigenvectors = qr_algorithm(A)

    print("Các trị riêng xấp xỉ:")
    print(eigenvalues)

    print("Các vectơ riêng xấp xỉ (theo cột):")
    print(eigenvectors)
