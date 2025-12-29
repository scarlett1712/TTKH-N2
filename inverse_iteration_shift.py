import numpy as np

def inverse_iteration_shift(A, mu, v0, tol=1e-8, max_iter=1000):
    n = A.shape[0]

    # Chuẩn hóa vector ban đầu
    v = v0 / np.linalg.norm(v0)
    I = np.eye(n)

    for k in range(max_iter):
        try:
            # Giải (A - μI) w = v
            w = np.linalg.solve(A - mu * I, v)
        except np.linalg.LinAlgError:
            print("Ma trận A - μI suy biến, dừng lặp.")
            break

        v_new = w / np.linalg.norm(w)

        # Kiểm tra hội tụ
        if abs(v_new @ v) > 1 - tol:
            print(f"Hội tụ tại bước lặp {k + 1}")
            v = v_new
            break

        v = v_new
    else:
        print("Không hội tụ trong số bước lặp cho phép.")

    # Rayleigh quotient cho trị riêng
    eigenvalue = v.T @ A @ v
    return eigenvalue, v


# ===================== TEST =====================
if __name__ == "__main__":
    A = np.array([
        [4, 1, 1],
        [1, 3, 0],
        [1, 0, 2]
    ], dtype=float)

    mu = 2.5                     # shift gần trị riêng cần tìm
    v0 = np.ones(A.shape[0])     # vector khởi tạo

    eigenvalue, eigenvector = inverse_iteration_shift(A, mu, v0)

    print("Trị riêng xấp xỉ (gần μ nhất):")
    print(eigenvalue)

    print("Vector riêng tương ứng:")
    print(eigenvector)