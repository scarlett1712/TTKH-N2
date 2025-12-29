import numpy as np

def rayleigh_quotient_iteration(A, v0, tol=1e-10, max_iter=100):
    n = A.shape[0]
    
    v = v0 / np.linalg.norm(v0)
    lam = v.T @ A @ v   # Rayleigh quotient ban đầu
    
    for k in range(max_iter):
        try:
            w = np.linalg.solve(A - lam * np.eye(n), v)
        except np.linalg.LinAlgError:
            print("Ma trận suy biến, dừng lặp.")
            break
        
        v = w / np.linalg.norm(w)
        lam_new = v.T @ A @ v
        
        if abs(lam_new - lam) < tol:
            print(f"Hội tụ tại bước lặp {k + 1}")
            lam = lam_new
            break
        
        lam = lam_new
    
    return lam, v


# Ví dụ sử dụng
if __name__ == "__main__":
    A = np.array([
        [4, 1, 1],
        [1, 3, 0],
        [1, 0, 2]
    ], dtype=float)

    v0 = np.ones(A.shape[0])   # vector khởi tạo

    eigenvalue, eigenvector = rayleigh_quotient_iteration(A, v0)

    print("Trị riêng xấp xỉ:")
    print(eigenvalue)

    print("Vector riêng tương ứng:")
    print(eigenvector)
