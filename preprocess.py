import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

REQUIRED_COLUMNS = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth'
]

def preprocess_data(df):
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        raise ValueError("Dữ liệu không chứa đầy đủ các cột cần thiết để phân cụm.")

    selected = df[REQUIRED_COLUMNS]

    imputer = SimpleImputer(strategy='median')
    imputed = imputer.fit_transform(selected)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled)

    return X_pca, df
