import numpy as np

def inverse_iteration(A, v0, tol=1e-8, max_iter=1000):
    n = A.shape[0]
    v = v0 / np.linalg.norm(v0)

    for k in range(max_iter):
        try:
            w = np.linalg.solve(A, v)
        except np.linalg.LinAlgError:
            print("Ma trận A suy biến, dừng lặp.")
            break

        v_new = w / np.linalg.norm(w)

        # Điều kiện hội tụ chuẩn (tránh đổi dấu)
        if abs(v_new @ v) > 1 - tol:
            print(f"Hội tụ tại bước lặp {k + 1}")
            v = v_new
            break

        v = v_new
    else:
        print("Không hội tụ trong số bước lặp cho phép.")

    eigenvalue = v.T @ A @ v
    return eigenvalue, v

if __name__ == "__main__":
    A = np.array([
        [4, 1, 1],
        [1, 3, 0],
        [1, 0, 2]
    ], dtype=float)

    v0 = np.ones(A.shape[0])

    eigenvalue, eigenvector = inverse_iteration(A, v0)

    print("Trị riêng nhỏ nhất xấp xỉ:")
    print(eigenvalue)

    print("Vector riêng tương ứng:")
    print(eigenvector)
