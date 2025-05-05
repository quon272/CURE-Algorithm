import streamlit as st
import pandas as pd
from preprocess import preprocess_data
from cure import simple_cure
from utils import plot_clusters, download_link
import io
from sklearn.metrics import silhouette_score

st.set_page_config(layout='wide')
st.title("📊 Phân cụm khách hàng bằng thuật toán CURE")

st.markdown("Tải lên file dữ liệu khách hàng (.csv, .tsv)")

uploaded_file = st.file_uploader("Drag and drop file here", type=['csv', 'tsv'])

if uploaded_file:
    # Đọc dữ liệu
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        st.stop()

    st.subheader("🔍 Xem trước dữ liệu")
    st.dataframe(df.head())

    # Sidebar: Tham số CURE
    st.sidebar.header("⚙️ Tham số CURE")
    n_clusters = st.sidebar.slider("Số cụm", 2, 10, 5)
    n_representatives = st.sidebar.slider("Số điểm đại diện mỗi cụm", 2, 10, 5)
    shrink_factor = st.sidebar.slider("Shrink factor", 0.0, 1.0, 0.2, step=0.05)

    try:
        X_pca, original_df = preprocess_data(df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    labels = simple_cure(X_pca, n_clusters, n_representatives, shrink_factor)

    st.subheader("📈 Kết quả phân cụm")
    fig = plot_clusters(X_pca, labels)
    st.pyplot(fig)

    # Tính toán Silhouette Coefficient
    try:
        silhouette_avg = silhouette_score(X_pca, labels)
        st.metric("Silhouette Coefficient", f"{silhouette_avg:.2f}")
    except Exception as e:
        st.error(f"Lỗi khi tính toán Silhouette Coefficient: {e}")

    result_df = original_df.copy()
    result_df["Cluster"] = labels
    st.dataframe(result_df.head())

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.markdown(download_link(csv_buffer.getvalue(), "clustered_customers.csv", "📥 Tải kết quả về"), unsafe_allow_html=True)
