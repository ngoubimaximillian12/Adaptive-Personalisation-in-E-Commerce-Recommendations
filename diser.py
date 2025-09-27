"""
Recommender Systems — Methodology → Implementation → Evaluation
================================================================
This single Python script mirrors your Methodology and Implementation chapters and
produces the Evaluation (RMSE/MAE + training curves) in a **simple, readable** way.

It covers:
  • Data loading, cleaning, and EDA-ready features
  • Feature engineering: Composite interaction score + 15‑day recency weighting
  • Content features via TF‑IDF on category text + TruncatedSVD (20 dims)
  • Train/test split (80/20, user‑stratified-ish) on implicit data converted to scores
  • Baselines (Surprise): SVD, SVD++, NMF → RMSE/MAE on test
  • Hybrid neural network with user/item embeddings + numeric features (MLP 256‑128‑64)
    with BatchNorm, Dropout, Adam + EarlyStopping/ReduceLROnPlateau → RMSE/MAE on test
  • Comparison chart and minimal, clear prints use this to run /Users/ngoubimaximilliandiamgha/PycharmProjects/PythonProject2/diser.py


HOW TO RUN
----------
1) Install deps (in your environment):
   pip install numpy pandas scikit-learn scikit-surprise matplotlib seaborn tensorflow==2.* tqdm

2) Set DATA_PATH below to your CSV (e.g., "2020-Jan.csv").

3) Run: python this_file.py    (or copy cells into a notebook)

NOTES
-----
• This script is designed to run on CPU with 16 GB RAM. If your CSV is very large (millions of rows), set
  SAMPLE_FRACTION < 1.0 or set N_USERS_MAX/N_ITEMS_MAX to cap the working set for development.
• All steps are explained in simple English inline comments for clarity.
"""

# ============================
# 0) Imports & Config
# ============================
from __future__ import annotations
import os, math, warnings, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# sklearn utilities
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Surprise (matrix factorisation baselines)
try:
    from surprise import Dataset, Reader, SVD, SVDpp, NMF
except Exception as e:
    Dataset = Reader = SVD = SVDpp = NMF = None
    print("[Note] surprise not found — install with: pip install scikit-surprise")

# Deep learning (hybrid model)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# Progress bars
from tqdm import tqdm

# ----------------------------
# Configurable parameters
# ----------------------------
DATA_PATH = "2020-Jan.csv"   # <-- change to your file path
TIME_COLUMN = "event_time"
USER_COL = "user_id"
ITEM_COL = "product_id"
EVENT_COL = "event_type"      # one of: view, cart, remove_from_cart, purchase
PRICE_COL = "price"           # if present; script handles if missing
CAT_COL = "category_code"     # text category path like "electronics.smartphone"
BRAND_COL = "brand"           # optional
SESSION_COL = "user_session"  # optional

# Sampling / shaping knobs for memory safety on modest hardware
SAMPLE_FRACTION = 1.0          # set <1.0 for quick dev runs
N_USERS_MAX = None             # e.g., 50_000 to cap users
N_ITEMS_MAX = None             # e.g., 50_000 to cap items

# Feature engineering knobs
HALF_LIFE_DAYS = 15.0          # recency half-life for exponential decay
COMPOSITE_WEIGHTS = {          # simple, interpretable implicit score
    "view": 0.10,
    "cart": 0.40,
    "remove_from_cart": -0.30,
    "purchase": 1.00,
}
TFIDF_NGRAM_RANGE = (1, 2)
SVD_N_COMPONENTS = 20          # for TF-IDF dimensionality reduction

# Surprise hyperparameters (as in your chapter)
SVD_FACTORS = 100
SVDPP_FACTORS = 100
NMF_FACTORS = 50
SVD_EPOCHS = 30
SVDPP_EPOCHS = 30
NMF_EPOCHS = 50
SVD_REG = 0.05
SVDPP_REG = 0.02

# NN hyperparameters (as in your chapter)
EMBED_DIM = 32
MLP_SIZES = [256, 128, 64]
DROPOUT = 0.3
BATCH_SIZE = 1024
EPOCHS = 50
PATIENCE = 5
LR_PATIENCE = 3
LR_FACTOR = 0.2

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def simple_print(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

# 1) Load Data & Basic Cleaning
# ============================
simple_print("1) Loading & basic cleaning")

# Point to your local file here (one place only)
DATA_PATH = "/Users/ngoubimaximilliandiamgha/Desktop/2020-Jan.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Set DATA_PATH to your CSV; not found: {DATA_PATH}")


# Parse dates on load for speed; non-existent columns are ignored safely below
parse_cols = [c for c in [TIME_COLUMN] if c]
df = pd.read_csv(DATA_PATH, low_memory=False)

# Optional sampling for dev speed
if SAMPLE_FRACTION < 1.0:
    df = df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_STATE).reset_index(drop=True)

print("Raw shape:", df.shape)

# Ensure datetime and simple time features
if TIME_COLUMN in df.columns:
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors='coerce')
    df["hour"] = df[TIME_COLUMN].dt.hour
    df["day_of_week"] = df[TIME_COLUMN].dt.day_name()
else:
    # create placeholders if time column missing
    df["hour"] = 0
    df["day_of_week"] = "Unknown"

# Basic fills to avoid NaNs later
if BRAND_COL in df.columns:
    df[BRAND_COL] = df[BRAND_COL].fillna("unknown")
if SESSION_COL in df.columns:
    df[SESSION_COL] = df[SESSION_COL].fillna("unknown")

# Clean price: drop negatives; fill missing with product median (if columns exist)
if PRICE_COL in df.columns:
    # replace absurd negatives (domain choice)
    df = df[df[PRICE_COL].isna() | (df[PRICE_COL] >= 0)]
    if df[PRICE_COL].isna().any():
        df[PRICE_COL] = df.groupby(ITEM_COL)[PRICE_COL].transform(lambda s: s.fillna(s.median()))

# Drop rows with missing core IDs
df = df.dropna(subset=[USER_COL, ITEM_COL])

# Optional caps on users/items to keep matrix sizes friendly
if N_USERS_MAX is not None:
    top_users = df[USER_COL].value_counts().nlargest(N_USERS_MAX).index
    df = df[df[USER_COL].isin(top_users)]
if N_ITEMS_MAX is not None:
    top_items = df[ITEM_COL].value_counts().nlargest(N_ITEMS_MAX).index
    df = df[df[ITEM_COL].isin(top_items)]

print("After cleaning shape:", df.shape)

# ============================
# 2) Feature Engineering (Simple English)
# ============================
simple_print("2) Feature engineering: composite score + recency + content")

# 2.1 Split category path into parts (e.g., electronics.smartphone → cat1, cat2, cat3)
if CAT_COL in df.columns and df[CAT_COL].notna().any():
    split = df[CAT_COL].astype(str).str.split('.', n=2, expand=True)
    for i, col in enumerate(["cat1","cat2","cat3"][:split.shape[1]]):
        df[col] = split[i]
else:
    df["cat1"] = "unknown"

# 2.2 Composite Interaction Score (views, carts, removes, purchases → one number)
#     We count how many of each event a user did for an item, weight them, then sum.
#     Simple English: "more purchases and carts push the score up; removes pull it down; views add a little".

# Map raw event labels to canonical keys used in COMPOSITE_WEIGHTS
EVENT_MAP = {
    "view": "view",
    "cart": "cart",
    "remove_from_cart": "remove_from_cart",
    "purchase": "purchase",
    # Allow common variants if present
    "add_to_cart": "cart",
    "remove": "remove_from_cart",
}

if EVENT_COL in df.columns:
    df["event_key"] = df[EVENT_COL].map(lambda x: EVENT_MAP.get(str(x), str(x)))
else:
    df["event_key"] = "view"

# Count per (user,item, event_key)
counts = (
    df.groupby([USER_COL, ITEM_COL, "event_key"], as_index=False)
      .size()
      .rename(columns={"size":"cnt"})
)
# Pivot to columns view/cart/remove_from_cart/purchase (fill 0s)
wide = counts.pivot_table(index=[USER_COL, ITEM_COL], columns="event_key", values="cnt", fill_value=0)
# Ensure all expected keys exist
for k in COMPOSITE_WEIGHTS.keys():
    if k not in wide.columns:
        wide[k] = 0
wide = wide.reset_index()

# Weighted sum
score = np.zeros(len(wide), dtype=float)
for k, w in COMPOSITE_WEIGHTS.items():
    score += w * wide[k].values
wide["raw_score"] = score

# 2.3 Recency Weighting (15-day half-life): newer interactions count more
#     Simple English: "recent actions matter more than old ones".
if TIME_COLUMN in df.columns:
    # last interaction time per (user,item)
    last_t = df.groupby([USER_COL, ITEM_COL])[TIME_COLUMN].max().reset_index().rename(columns={TIME_COLUMN:"last_time"})
    wide = wide.merge(last_t, on=[USER_COL, ITEM_COL], how="left")
    # days since last interaction relative to dataset max time
    max_t = pd.to_datetime(df[TIME_COLUMN]).max()
    days_since = (max_t - wide["last_time"]).dt.total_seconds() / (3600*24)
    # Exponential decay: w = 0.5 ** (days / half_life)
    recency_w = np.power(0.5, days_since.fillna(days_since.max()) / HALF_LIFE_DAYS)
else:
    recency_w = 1.0
wide["recency_weight"] = recency_w
wide["score_recency"] = wide["raw_score"] * wide["recency_weight"]

# 2.4 Normalize score to [0,1] for comparability across models
min_s, max_s = wide["score_recency"].min(), wide["score_recency"].max()
if max_s > min_s:
    wide["score_norm"] = (wide["score_recency"] - min_s) / (max_s - min_s)
else:
    wide["score_norm"] = 0.0

# 2.5 Content features via TF‑IDF(SVD) on category text (brand + category path)
if CAT_COL in df.columns or BRAND_COL in df.columns:
    prod = df[[ITEM_COL]].drop_duplicates().copy()
    cat_text = df.groupby(ITEM_COL).agg({
        CAT_COL: lambda s: ' '.join(sorted(set([str(x) for x in s if pd.notna(x)]))),
        BRAND_COL: lambda s: ' '.join(sorted(set([str(x) for x in s if pd.notna(x)]))),
    }) if (CAT_COL in df.columns and BRAND_COL in df.columns) else (
        df.groupby(ITEM_COL).agg({CAT_COL: lambda s: ' '.join(sorted(set([str(x) for x in s if pd.notna(x)])))})
    )
    cat_text = cat_text.fillna("")
    cat_text["text"] = (cat_text.get(CAT_COL, "") + " " + cat_text.get(BRAND_COL, "")).str.strip()
    vectorizer = TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE, min_df=2)
    tfidf = vectorizer.fit_transform(cat_text["text"]) if len(cat_text) > 0 else None
    if tfidf is not None and tfidf.shape[1] > 0:
        svd = TruncatedSVD(n_components=min(SVD_N_COMPONENTS, tfidf.shape[1]-1) if tfidf.shape[1] > 1 else 1,
                           random_state=RANDOM_STATE)
        dense = svd.fit_transform(tfidf)
        tfidf_df = pd.DataFrame(dense, index=cat_text.index)
        tfidf_df.columns = [f"tfidf_{i:02d}" for i in range(tfidf_df.shape[1])]
        item_features = tfidf_df.reset_index()
    else:
        item_features = pd.DataFrame({ITEM_COL: df[ITEM_COL].unique()})
else:
    item_features = pd.DataFrame({ITEM_COL: df[ITEM_COL].unique()})

# Merge engineered score with item content features
ratings = wide.merge(item_features, on=ITEM_COL, how="left")

# ============================
# 3) Train/Test Split (80/20, stratify by user where possible)
# ============================
simple_print("3) Train/test split (80/20)")
# Label-encode users/items for models that need integer IDs
le_user = LabelEncoder(); le_item = LabelEncoder()
ratings["user_enc"] = le_user.fit_transform(ratings[USER_COL].astype(str))
ratings["item_enc"] = le_item.fit_transform(ratings[ITEM_COL].astype(str))

# Basic train/test split per interaction row. We try to preserve user distribution with stratify on user_enc
try:
    X_train, X_test = train_test_split(ratings, test_size=0.2, random_state=RANDOM_STATE, stratify=ratings["user_enc"])
except ValueError:
    # Fallback if some users have only one interaction (can't stratify properly)
    X_train, X_test = train_test_split(ratings, test_size=0.2, random_state=RANDOM_STATE)

print("Train size:", X_train.shape, "Test size:", X_test.shape)

# Target for all models: the normalized implicit score (0..1)
y_train = X_train["score_norm"].values.astype(np.float32)
y_test  = X_test["score_norm"].values.astype(np.float32)

# ============================
# 4) Baselines — SVD / SVD++ / NMF (Surprise)
# ============================
simple_print("4) Baselines (Surprise): SVD / SVD++ / NMF → RMSE & MAE")
results = []

if Dataset is None:
    print("Surprise not installed; skipping MF baselines.")
else:
    def eval_surprise(algo, name: str):
        # Build Dataset from train
        reader = Reader(rating_scale=(0.0, 1.0))
        data_train = Dataset.load_from_df(X_train[[USER_COL, ITEM_COL, "score_norm"]], reader).build_full_trainset()
        algo.fit(data_train)
        # Predict on test set
        preds = []
        for _, row in X_test.iterrows():
            preds.append(algo.predict(str(row[USER_COL]), str(row[ITEM_COL])).est)
        preds = np.array(preds, dtype=float)
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        print(f"{name:6s}  RMSE={rmse:.4f}  MAE={mae:.4f}")
        results.append({"model": name, "RMSE": rmse, "MAE": mae})

    eval_surprise(SVD(n_factors=SVD_FACTORS, n_epochs=SVD_EPOCHS, reg_all=SVD_REG, random_state=RANDOM_STATE), "SVD")
    eval_surprise(SVDpp(n_factors=SVDPP_FACTORS, n_epochs=SVDPP_EPOCHS, reg_all=SVDPP_REG, random_state=RANDOM_STATE), "SVD++")
    eval_surprise(NMF(n_factors=NMF_FACTORS, n_epochs=NMF_EPOCHS, random_state=RANDOM_STATE), "NMF")

# ============================
# 5) Hybrid Neural Network with Embeddings + Numeric Features
# ============================
simple_print("5) Hybrid neural network (embeddings + MLP) → RMSE & MAE")

# Inputs:
#   • user_enc (int id) → user embedding
#   • item_enc (int id) → item embedding
#   • numeric content features (tfidf_*) → scaled and fed to MLP

num_cols = [c for c in ratings.columns if c.startswith("tfidf_")]
if len(num_cols) == 0:
    # If you have no content features, create a tiny placeholder numeric col
    ratings["num_dummy"] = 0.0
    num_cols = ["num_dummy"]

# Align train/test numeric matrices
Xnum_train = X_train[num_cols].fillna(0.0).values
Xnum_test  = X_test[num_cols].fillna(0.0).values

# Standardize numeric features (as per your chapter)
scaler = StandardScaler()
Xnum_train = scaler.fit_transform(Xnum_train)
Xnum_test  = scaler.transform(Xnum_test)

n_users = ratings["user_enc"].max() + 1
n_items = ratings["item_enc"].max() + 1

# Build model
# Simple English architecture:
#   1) Look up a small vector (embedding) for the user and the item
#   2) Concatenate those vectors with the numeric content features
#   3) Pass through 3 dense layers with BatchNorm + Dropout to learn non‑linear patterns
#   4) Output a single number in [0,1] (the predicted score)

# Inputs
inp_user = layers.Input(shape=(1,), name="user_enc")
inp_item = layers.Input(shape=(1,), name="item_enc")
inp_num  = layers.Input(shape=(len(num_cols),), name="num_features")

# Embeddings
emb_user = layers.Embedding(input_dim=n_users, output_dim=EMBED_DIM, name="user_emb")(inp_user)
emb_item = layers.Embedding(input_dim=n_items, output_dim=EMBED_DIM, name="item_emb")(inp_item)

# Flatten embeddings
emb_user = layers.Flatten()(emb_user)
emb_item = layers.Flatten()(emb_item)

# Combine all features
x = layers.Concatenate()([emb_user, emb_item, inp_num])

# MLP stack
for i, h in enumerate(MLP_SIZES):
    x = layers.Dense(h, activation="relu", name=f"dense_{i+1}")(x)
    x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
    x = layers.Dropout(DROPOUT, name=f"drop_{i+1}")(x)

# Output in [0,1]
out = layers.Dense(1, activation="sigmoid", name="score")(x)

model = models.Model(inputs=[inp_user, inp_item, inp_num], outputs=out)
model.compile(optimizer=optimizers.Adam(), loss="mse")  # MSE per your chapter

# Callbacks: EarlyStopping + ReduceLROnPlateau
cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", patience=LR_PATIENCE, factor=LR_FACTOR)
]

# Prepare model inputs
u_train = X_train["user_enc"].values
i_train = X_train["item_enc"].values
u_test  = X_test["user_enc"].values
i_test  = X_test["item_enc"].values

hist = model.fit(
    x=[u_train, i_train, Xnum_train],
    y=y_train,
    validation_data=([u_test, i_test, Xnum_test], y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    callbacks=cbs
)

# Evaluate
pred_nn = model.predict([u_test, i_test, Xnum_test], verbose=0).reshape(-1)
rmse_nn = math.sqrt(mean_squared_error(y_test, pred_nn))
mae_nn  = mean_absolute_error(y_test, pred_nn)
print(f"Neural  RMSE={rmse_nn:.4f}  MAE={mae_nn:.4f}")
results.append({"model": "Neural (Hybrid)", "RMSE": rmse_nn, "MAE": mae_nn})

# ============================
# 6) Visualize: Training curve + RMSE/MAE comparison
# ============================
simple_print("6) Plots: training curve + model comparison")
fig1 = plt.figure(figsize=(6,4))
plt.plot(hist.history["loss"], label="train")
plt.plot(hist.history["val_loss"], label="val")
plt.title("Neural model — loss curve (MSE)")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
plt.show()

if len(results) > 0:
    df_res = pd.DataFrame(results).sort_values("RMSE")
    fig2 = plt.figure(figsize=(6,4))
    sns.barplot(data=df_res, x="model", y="RMSE")
    plt.title("RMSE by model (lower is better)")
    plt.xticks(rotation=20)
    plt.tight_layout(); plt.show()

    fig3 = plt.figure(figsize=(6,4))
    sns.barplot(data=df_res, x="model", y="MAE")
    plt.title("MAE by model (lower is better)")
    plt.xticks(rotation=20)
    plt.tight_layout(); plt.show()

# ============================
# 7) Minimal EDA helpers (optional)
# ============================
simple_print("7) Optional EDA: event type distribution & activity heatmap")
if EVENT_COL in df.columns:
    plt.figure(figsize=(6,4))
    df[EVENT_COL].value_counts().plot(kind='bar')
    plt.title('Event Type Distribution')
    plt.xlabel('Event Type'); plt.ylabel('Count'); plt.tight_layout(); plt.show()

if TIME_COLUMN in df.columns:
    heat = df.pivot_table(index='day_of_week', columns='hour', values=USER_COL, aggfunc='count').fillna(0)
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    heat = heat.reindex(order)
    plt.figure(figsize=(10,4))
    sns.heatmap(heat, cmap='YlGnBu')
    plt.title('User Activity Heatmap by Hour and Day'); plt.tight_layout(); plt.show()

# ============================
# 8) Final prints & cleanup
# ============================
simple_print("8) Done — results table")
if len(results) > 0:
    print(pd.DataFrame(results))
else:
    print("No results to show (likely Surprise baselines were skipped). Neural results printed above.")

# Optionally save the neural model and encoders for later inference
# model.save('hybrid_recommender.h5')
# pd.DataFrame({"user": le_user.classes_}).to_csv("users.csv", index=False)
# pd.DataFrame({"item": le_item.classes_}).to_csv("items.csv", index=False)

# Free memory
del df, ratings, X_train, X_test; gc.collect()
print("All done.")
