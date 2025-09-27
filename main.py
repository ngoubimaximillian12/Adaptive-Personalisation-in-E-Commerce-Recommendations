import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from surprise import Dataset, Reader, SVDpp, SVD, NMF, accuracy

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
from tqdm.notebook import tqdm
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/Users/ngoubimaximilliandiamgha/Desktop/2020-Jan.csv')
print('Raw shape:', df.shape)
display(df.dtypes, df.describe())


df['event_time']    = pd.to_datetime(df['event_time'])
df['hour']          = df['event_time'].dt.hour
df['day_of_week']   = df['event_time'].dt.day_name()

# Split category_code
if 'category_code' in df.columns and df['category_code'].notna().any():
    df[['cat1', 'cat2', 'cat3']] = df['category_code'].str.split('.', expand=True, n=2)
df['brand'] = df['brand'].fillna('unknown')
df['user_session'] = df['user_session'].fillna('unknown')

# Filter views and buys
df_views = df[df['event_type'] == 'view']
df_buys = df[df['event_type'] == 'purchase']

# Merge view and buy to see conversion
view_buy = pd.merge(df_views, df_buys, on=['user_id', 'product_id'], how='left', suffixes=('_view', '_buy'))
view_buy['converted'] = view_buy['event_type_buy'].notna().astype(int)

# Aggregate user-product interactions
interaction_counts = df.groupby(['user_id', 'product_id']).size().reset_index(name='interaction_count')

# Popularity-based recommendations
popular_products = df['product_id'].value_counts().head(10).index.tolist()

# Label Encoding
le_user = LabelEncoder()
le_product = LabelEncoder()

df['user_id_enc'] = le_user.fit_transform(df['user_id'])
df['product_id_enc'] = le_product.fit_transform(df['product_id'])

# Collaborative Filtering using Surprise
reader = Reader(rating_scale=(1, 10))
interaction_df = interaction_counts.copy()
interaction_df['rating'] = np.clip(interaction_df['interaction_count'], 1, 10)
data = Dataset.load_from_df(interaction_df[['user_id', 'product_id', 'rating']], reader)
trainset = data.build_full_trainset()
algo = SVDpp()
algo.fit(trainset)

# Predict for a specific user
uid = str(df['user_id'].iloc[0])
iid = str(df['product_id'].iloc[0])
pred = algo.predict(uid, iid)
print(f"Predicted rating for user {uid} and item {iid}: {pred.est:.2f}")

# Content-based filtering using TF-IDF
product_texts = df[['product_id', 'category_code']].dropna().drop_duplicates()
product_texts['text'] = product_texts['category_code'].fillna('')

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(product_texts['text'])

# Dimensionality Reduction
svd = TruncatedSVD(n_components=5)
reduced_features = svd.fit_transform(tfidf_matrix)

# Create a DataFrame of reduced features
product_features = pd.DataFrame(reduced_features, index=product_texts['product_id'])

# Deep Learning Recommender
user_product_matrix = df.pivot_table(index='user_id_enc', columns='product_id_enc',
                                     values='event_type', aggfunc='count', fill_value=0)

X = user_product_matrix.values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(X_train.shape[1], activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.2),
    TqdmCallback(verbose=1)
]

history = model.fit(
    X_train, X_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_val, X_val),
    callbacks=callbacks,
    verbose=0
)

# Recommend for a new user
user_index = 0
user_vector = X_val[user_index].reshape(1, -1)
predicted_vector = model.predict(user_vector)[0]

recommended_indices = predicted_vector.argsort()[-10:][::-1]
recommended_product_ids = user_product_matrix.columns[recommended_indices]

print("Recommended Product IDs:", recommended_product_ids.tolist())
# Visualizations
plt.figure(figsize=(10, 5))
df['event_type'].value_counts().plot(kind='bar')
plt.title('Event Type Distribution')
plt.xlabel('Event Type')
plt.ylabel('Count')
plt.show()

# Heatmap of interactions by hour and day
heatmap_data = df.pivot_table(index='day_of_week', columns='hour', values='user_id', aggfunc='count')
heatmap_data = heatmap_data.fillna(0)

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu')
plt.title('User Activity Heatmap by Hour and Day')
plt.show()

# Display top categories
top_categories = df['category_code'].value_counts().head(10)
plt.figure(figsize=(10, 5))
top_categories.plot(kind='bar')
plt.title('Top 10 Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Check brand distribution
top_brands = df['brand'].value_counts().head(10)
plt.figure(figsize=(10, 5))
top_brands.plot(kind='bar')
plt.title('Top 10 Brands')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
# User-level analysis
user_activity = df.groupby('user_id')['event_type'].value_counts().unstack().fillna(0)
user_activity['total'] = user_activity.sum(axis=1)
user_activity = user_activity.sort_values(by='total', ascending=False).head(10)

user_activity.drop(columns='total').plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Top 10 Active Users by Event Type')
plt.xlabel('User ID')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Event Type')
plt.tight_layout()
plt.show()

# Session analysis
session_counts = df['user_session'].value_counts().head(10)
plt.figure(figsize=(10, 4))
session_counts.plot(kind='bar')
plt.title('Top 10 Sessions by Activity')
plt.xlabel('Session ID')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Day of week activity
plt.figure(figsize=(10, 4))
df['day_of_week'].value_counts().loc[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']].plot(kind='bar')
plt.title('User Activity by Day of the Week')
plt.xlabel('Day')
plt.ylabel('Activity Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Hourly activity
plt.figure(figsize=(10, 4))
df['hour'].value_counts().sort_index().plot(kind='bar')
plt.title('User Activity by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Activity Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Save model if needed
model.save('recommender_model.h5')

# Export encoded data (optional)
encoded_df = df[['user_id', 'user_id_enc', 'product_id', 'product_id_enc']]
encoded_df.drop_duplicates().to_csv('encoded_user_product.csv', index=False)

print("Model saved and encoded data exported.")
