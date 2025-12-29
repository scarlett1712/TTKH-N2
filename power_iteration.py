import numpy as np

def power_iteration(A, tol=1e-6, max_iter=1000):
    n = A.shape[0]

    # Khởi tạo vector ban đầu (không được là vector 0)
    v = np.random.rand(n)
    v /= np.linalg.norm(v)

    eigenvalue_old = 0.0

    for k in range(max_iter):
        # Nhân ma trận
        w = A @ v

        # Chuẩn hóa
        v = w / np.linalg.norm(w)

        # Rayleigh quotient
        eigenvalue = v.T @ A @ v

        # Kiểm tra hội tụ
        if abs(eigenvalue - eigenvalue_old) < tol:
            print(f"Hội tụ tại bước lặp {k + 1}")
            break

        eigenvalue_old = eigenvalue
    else:
        print("Không hội tụ trong số bước lặp cho phép.")

    return eigenvalue, v


# ===================== TEST =====================
if __name__ == "__main__":
    A = np.array([
        [4, 1, 1],
        [1, 3, 0],
        [1, 0, 2]
    ], dtype=float)

    eigenvalues, eigenvectors = power_iteration(A)

    print("Các trị riêng xấp xỉ:")
    print(eigenvalues)

    print("Các vectơ riêng xấp xỉ (theo cột):")
    print(eigenvectors)
