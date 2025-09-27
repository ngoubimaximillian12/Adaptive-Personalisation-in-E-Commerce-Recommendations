# Adaptive Personalisation in E‑Commerce Recommendations

> Powerful, production‑ready hybrid recommender system with classic CF baselines and deep learning models. Clean API, strong evaluation tools, and battle‑tested troubleshooting.

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python"></img>
  <img src="https://img.shields.io/badge/TensorFlow-2.13-orange" alt="TensorFlow"></img>
  <img src="https://img.shields.io/badge/Scikit--Surprise-latest-lightgrey" alt="Surprise"></img>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success" alt="Status"></img>
</p>

---

## ✨ Features

* **Hybrid recommender**: matrix factorisation + neural embeddings.
* **Baselines included**: SVD, SVD++, NMF via `surprise`.
* **Deep model**: embeddings + MLP with batch normalization for faster, more stable convergence.
* **Implicit & explicit feedback** support.
* **Comprehensive evaluation**: RMSE, MAE, ranking metrics (precision@k, recall@k, MAP, NDCG), learning curves.
* **EDA & visualisations** to understand users, items, and interactions.
* **Troubleshooting cookbook** for memory, performance, and dependency issues.

> [!TIP]
> Looking for a quick run? Jump to **[🚀 Quick Start](#-quick-start)**.

---

## 📚 Table of Contents

* [Features](#-features)
* [Architecture](#-architecture)
* [Models](#-models)
* [Data Format](#-data-format)
* [Installation](#-installation)
* [Quick Start](#-quick-start)
* [Configuration](#-configuration)
* [Training](#-training)
* [Evaluation & Reports](#-evaluation--reports)
* [Troubleshooting](#-troubleshooting)
* [References](#-references)
* [Contributing](#-contributing)
* [License](#-license)
* [Author](#-author)
* [Changelog](#-changelog)

---

## 🧱 Architecture

```
CSV → preprocessing → split (train/valid/test) →
  ├─ surprise baselines (SVD/SVD++/NMF)
  ├─ neural recommender (embeddings + MLP + BN)
  └─ hybrid ensembling (optional)
→ metrics (RMSE/MAE/ranking) → reports & charts
```

**Batch normalization** is used throughout the MLP to stabilise activations and speed up convergence.

> [!NOTE]
> This repo targets **TensorFlow 2.13** and **scikit‑surprise**. See [Missing Dependencies](#missing-dependencies) for exact pins.

---

## 🧠 Models

* **Matrix Factorisation (MF)** — SVD / SVD++ / NMF via `surprise`.
* **Neural Recommender** — user/item embeddings → concatenation → MLP (ReLU + BatchNorm + Dropout).
* **Hybrid** — weighted average or stacking of MF & neural scores.

---

## 🗂️ Data Format

Input CSV (minimum columns):

```csv
user_id,product_id,event_type,event_time,price,category_code,brand
1234,SKU-001,view,2024-07-12T10:15:00,19.99,apparel.shoes,Nike
```

* `event_type` examples: `view`, `cart`, `purchase`. Map these to implicit strengths if needed.
* Timestamps should be ISO‑8601 or parseable by `pandas.to_datetime`.

---

## 🛠 Installation

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 2) Install core dependencies
pip install numpy pandas scikit-learn scikit-surprise tensorflow==2.13.0 matplotlib seaborn tqdm
```

> [!WARNING]
> GPU users: ensure your CUDA/cuDNN stack matches TensorFlow 2.13 requirements.

---

## 🚀 Quick Start

```bash
# Prepare your data file (CSV)
# user_id,product_id,event_type,event_time,price,category_code,brand

# Run the pipeline
python diser.py
```

**Output includes:**

* RMSE & MAE per model
* Training curves
* Model comparison charts
* EDA visualisations

**Expected runtime:** 10–30 minutes (data/hardware dependent).
**Memory usage:** 2–8 GB depending on dataset size.

---

## ⚙️ Configuration

Tweak these constants to balance speed, accuracy, and resource use.

```python
# ===== Data downsampling (controls memory/fit speed) =====
SAMPLE_FRACTION = 0.1
N_USERS_MAX = 5000
N_ITEMS_MAX = 5000

# ===== Neural model capacity =====
EMBED_DIM = 64
MLP_SIZES = [512, 256, 128]
EPOCHS = 100
```

> [!TIP]
> Start small, confirm the pipeline works, then scale up users/items and epochs.

---

## 🏋️ Training

The neural model uses **Batch Normalization** to improve stability and speed up convergence.

```text
Batch normalization: Faster convergence and stability
```

You can enable/disable BN or adjust dropout in the model config.

---

## 📈 Evaluation & Reports

* **Regression metrics**: RMSE, MAE (for explicit ratings or calibrated scores).
* **Ranking metrics**: precision@k, recall@k, MAP, NDCG (for top‑N recommendation).
* **Visuals**: learning curves, score distributions, feature histograms.

Reports are written to the `reports/` folder (plots as PNGs).

---

## 🔧 Troubleshooting

### Common Issues

#### Memory Errors

```python
# Reduce dataset size
SAMPLE_FRACTION = 0.1
N_USERS_MAX = 5000
N_ITEMS_MAX = 5000
```

#### Poor Performance

```python
# Increase model complexity
EMBED_DIM = 64
MLP_SIZES = [512, 256, 128]
EPOCHS = 100
```

#### Missing Dependencies

```bash
pip install --upgrade scikit-surprise
pip install tensorflow==2.13.0
```

> [!IMPORTANT]
> Always check your Python version and virtual environment. Conflicts often come from mixed installations.

---

## 📚 References

### Academic Papers

* **Matrix Factorization Techniques for Recommender Systems** — Koren et al., 2009
* **Deep Learning for Recommender Systems** — Zhang et al., 2019
* **Implicit Feedback Datasets for Collaborative Filtering** — Hu et al., 2008

### Libraries and Frameworks

* **Surprise** — Python scikit for recommender systems
* **TensorFlow** — Deep learning framework
* **Scikit‑learn** — Machine learning library

---

## 🤝 Contributing

We welcome contributions! Please:

1. **Fork** the repository
2. **Create** a feature branch

   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit** with a clear message

   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push** your branch

   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request** with a concise description and screenshots where helpful.

### Development Guidelines

* Follow **PEP 8** style
* Add **comprehensive docstrings**
* Include **unit tests** for new features
* Update **documentation** for API changes

---

## 📄 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## 👨‍💻 Author

**Ngoubi Maximillian Diamgha**
GitHub: [@ngoubimaximillian12](https://github.com/ngoubimaximillian12)
Email: [ngoubimaximilliandiangha@gmail.com](mailto:ngoubimaximilliandiangha@gmail.com)
LinkedIn: [https://www.linkedin.com/in/diangha-ngoubi-42a49b281/](https://www.linkedin.com/in/diangha-ngoubi-42a49b281/)

Website: *add your website link here*

---

## 🙏 Acknowledgments

* Surprise team for excellent collaborative filtering tools
* TensorFlow team for the deep learning framework
* E‑commerce research community for inspiration and best practices
* Open source contributors who made this project possible

---

## 📋 Changelog

### Version 1.0.0 — 2024‑09‑27

* Initial release
* Hybrid recommender system implementation
* Baseline models (SVD, SVD++, NMF)
* Neural network with embeddings
* Comprehensive evaluation framework
* EDA and visualisation tools

---

## 📦 Project Structure (suggested)

```
.
├── diser.py                 # Main entry point
├── models/                  # Model definitions (neural & wrappers)
├── data/                    # Raw/processed data (gitignored)
├── configs/                 # Config files
├── reports/                 # Metrics & charts output
├── notebooks/               # EDA & experiments
├── tests/                   # Unit tests
├── README.md
└── LICENSE
```

---

## ❓ FAQ

**Q:** How do I switch to implicit feedback?
**A:** Map `event_type` to strengths (e.g., view=1, cart=3, purchase=5), then train on weighted interactions and evaluate with ranking metrics.

**Q:** Can I run without GPUs?
**A:** Yes. Start with small `N_USERS_MAX/N_ITEMS_MAX` and fewer epochs.

---

> For questions, issues, or contributions, please open an **Issue** or **PR** on the GitHub repository.
