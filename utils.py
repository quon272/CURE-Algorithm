import matplotlib.pyplot as plt
import base64

def plot_clusters(X_pca, labels):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', alpha=0.7)
    ax.set_title("Phân cụm khách hàng bằng CURE (PCA 2D)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True)
    return fig

def download_link(csv_str, filename, link_text):
    b64 = base64.b64encode(csv_str.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href
