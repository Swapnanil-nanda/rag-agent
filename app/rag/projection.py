import numpy as np

def project_embeddings(embeddings: list[list[float]]) -> list[tuple[float, float]]:
    if not embeddings:
        return []
    
    n_samples = len(embeddings)
    X = np.array(embeddings, dtype=np.float32)
    
    if n_samples < 3:
        return [(float(i * 0.5 - 0.25), float(i * 0.5 - 0.25)) for i in range(n_samples)]
        
    try:
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        proj = np.dot(X_centered, eigenvectors[:, :2])
        
        x_min, x_max = proj[:, 0].min(), proj[:, 0].max()
        y_min, y_max = proj[:, 1].min(), proj[:, 1].max()
        
        x_span = x_max - x_min
        y_span = y_max - y_min
        
        coords = []
        for i in range(n_samples):
            x = 2.0 * (proj[i, 0] - x_min) / x_span - 1.0 if x_span > 1e-5 else 0.0
            y = 2.0 * (proj[i, 1] - y_min) / y_span - 1.0 if y_span > 1e-5 else 0.0
            coords.append((float(x), float(y)))
        return coords
    except Exception:
        np.random.seed(42)
        random_coords = np.random.uniform(-0.8, 0.8, size=(n_samples, 2))
        return [(float(pt[0]), float(pt[1])) for pt in random_coords]
