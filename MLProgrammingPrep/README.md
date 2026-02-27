# The Student Data Journey
## Maplewood High School Analytics Project

A two-part Jupyter notebook series that walks through the complete data engineering and machine learning pipeline — from raw, messy data through to a working grade-prediction model. Built as a teaching resource for introductory data science and ML courses.

---

## Notebooks

### Part 1 — Understanding, Exploring, and Cleaning
`Part1_Understanding_and_Cleaning.ipynb`

Covers the first half of the data pipeline on a synthetic 500-student dataset:

| Topic | What You'll Do |
|-------|---------------|
| Pattern Recognition | Scatter plots revealing grade trends before any formal analysis |
| Supervised vs Unsupervised Learning | Side-by-side conceptual comparison with examples |
| Data Engineering Lifecycle | The 6-stage pipeline from raw data to insight |
| Attributes / Features / Objects | Identify numerical vs categorical features |
| Descriptive Statistics | Mean, median, mode, variance, std dev, quartiles, IQR — manually and with pandas |
| Missing Data (MNAR / MCAR / MAR) | Inject and handle all three types with appropriate strategies |
| Outliers | Detect with the IQR method; handle via winsorization (capping) |
| Sampling | SRS with/without replacement; stratified sampling to preserve group proportions |
| Regular Expressions | Extract student IDs, emails, and phone numbers from three inconsistent text formats |
| Data Reduction by Aggregation | Compress 500 rows into group-level summaries |

---

### Part 2 — Transforming, Reducing, and Modelling
`Part2_Transforming_and_Modeling.ipynb`

Picks up with the clean dataset and prepares it for machine learning:

| Topic | What You'll Do |
|-------|---------------|
| Min-Max Normalisation | Scale all numeric features to [0, 1] |
| Z-Score Normalisation | Standardise features to mean=0, std=1 |
| Box-Cox Transformation | Reduce skewness in right-skewed features |
| Encoding | Label encoding for ordinal categories; one-hot encoding for nominal categories |
| Discretization | Equal-width bins, equal-frequency bins, and custom letter-grade bins |
| Concept Hierarchies | Build grade → letter → pass/fail → achievement generalisation chains |
| K-Means Clustering | Group students into natural clusters without using labels (unsupervised) |
| Similarity Measures | Euclidean and Manhattan distance between data points |
| PCA | Compress 4 features into 2 principal components (~80% variance retained) |
| Train / Validation / Test Split | Stratified 70/15/15 split for honest model evaluation |
| Linear Regression | Build, train, and evaluate a grade-prediction model |

---

## Dataset

Both notebooks use a **synthetic** dataset of 500 Maplewood High School students generated with `numpy.random.seed(42)` — fully reproducible, no real student data. Part 2 rebuilds the dataset independently so each notebook is self-contained.

**Features:**

| Column | Type | Description |
|--------|------|-------------|
| `student_id` | string | Unique identifier (S0001–S0500) |
| `study_hours` | float | Hours studied per week (0–12) |
| `attendance_pct` | float | Percentage of classes attended (0–100) |
| `socio_score` | float | Socioeconomic score (1–10) |
| `prev_gpa` | float | Previous semester GPA (0–4) |
| `gender` | string | Male / Female |
| `parent_education` | string | No HS / High School / Some College / Bachelor's / Graduate |
| `extracurricular` | string | Yes / No |
| `final_grade` | float | Target variable — final grade percentage (0–100) |
| `teacher_comment` | string | Messy free-text with embedded IDs, emails, and phone numbers (Part 1 only) |

---

## Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies

---

## Setup

```bash
# 1. Clone or download the project
cd MLProgrammingPrep

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

Then open `Part1_Understanding_and_Cleaning.ipynb` and run cells top to bottom.
Follow up with `Part2_Transforming_and_Modeling.ipynb`.

---

## Key Libraries

| Library | Used For |
|---------|----------|
| `numpy` | Array operations, random data generation |
| `pandas` | DataFrames, groupby, encoding, discretization |
| `matplotlib` | All charts and subplots |
| `seaborn` | Statistical charts (heatmaps, styled histograms) |
| `scipy` | Box-Cox transformation, skewness calculation |
| `scikit-learn` | Scaling, PCA, K-Means, Linear Regression, train/test split, metrics |

---

## Learning Outcomes

By the end of both notebooks, you will be able to:

1. Describe the 6-stage data engineering lifecycle
2. Distinguish between supervised and unsupervised learning with concrete examples
3. Identify and handle MNAR, MCAR, and MAR missing data correctly
4. Detect outliers using the IQR method and apply winsorization
5. Choose the right normalisation technique (Min-Max, Z-Score, Box-Cox) for a given distribution
6. Encode categorical variables using label encoding and one-hot encoding
7. Discretize continuous values into meaningful bins
8. Build and interpret a concept hierarchy
9. Apply K-Means clustering and profile the resulting groups
10. Use PCA to reduce dimensionality and visualise high-dimensional data
11. Split data correctly into train/validation/test sets and avoid data leakage
12. Train a Linear Regression model and interpret R² and MAE
