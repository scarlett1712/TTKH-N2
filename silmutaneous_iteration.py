import numpy as np

def simultaneous_iteration(A, p, tol=1e-10, max_iter=200):
    n = A.shape[0]

    # 1. Khởi tạo không gian con ngẫu nhiên và trực chuẩn hóa
    V = np.random.randn(n, p)
    Q, _ = np.linalg.qr(V)

    for k in range(max_iter):
        # 2. Nhân ma trận
        Z = A @ Q

        # 3. Tái trực giao
        Q_new, _ = np.linalg.qr(Z)

        # 4. Ma trận Rayleigh
        T = Q_new.T @ A @ Q_new

        # 5. Kiểm tra hội tụ (ngoài đường chéo)
        off_diag_norm = np.linalg.norm(
            T - np.diag(np.diag(T)), ord='fro'
        )

        if off_diag_norm < tol:
            print(f"Hội tụ tại bước lặp {k + 1}")
            Q = Q_new
            break

        Q = Q_new
    else:
        print("Không hội tụ trong số bước lặp cho phép.")

    # 6. Trích xuất trị riêng & vector riêng
    eigenvalues = np.diag(Q.T @ A @ Q)
    eigenvectors = Q

    # Sắp xếp theo độ lớn giảm dần
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

    p = 3

    eigenvalues, eigenvectors = simultaneous_iteration(A,p)

    print("Các trị riêng xấp xỉ:")
    print(eigenvalues)

    print("Các vectơ riêng xấp xỉ (theo cột):")
    print(eigenvectors)
