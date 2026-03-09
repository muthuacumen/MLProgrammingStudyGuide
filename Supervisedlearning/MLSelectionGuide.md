# Machine Learning Model Selection Guide

### A Graduate-Level Decision Framework for Supervised Learning

> **How to use this guide:** Start at **Step 0** to determine your problem type (Regression vs Classification),
> then follow the decision tree for your branch. Each algorithm includes real-world examples,
> when to use it, and when to avoid it. Pros/Cons reference cards are at the end.

---

## Table of Contents

1. [Step 0: Regression or Classification?](#step-0-regression-or-classification)
2. [Regression Decision Tree](#-regression-decision-tree)
3. [Classification Decision Tree](#-classification-decision-tree)
4. [Algorithm Deep-Dive Cards](#-algorithm-deep-dive-cards)
   - [Regression Algorithms](#regression-algorithms)
   - [Classification Algorithms](#classification-algorithms)
5. [Universal Quick-Reference Table](#-universal-quick-reference-table)
6. [Common Mistakes to Avoid](#-common-mistakes-to-avoid)

---

## Step 0: Regression or Classification?

Before choosing ANY model, answer one question:

```
What does your target variable (y) look like?
│
├── It's a CONTINUOUS NUMBER (price, temperature, salary, score)
│   └── ➜ You have a REGRESSION problem         ──▶  Go to Section 2
│
├── It's a CATEGORY / LABEL (spam/not-spam, cat/dog/bird, pass/fail)
│   └── ➜ You have a CLASSIFICATION problem      ──▶  Go to Section 3
│
└── Not sure?
    ├── Can you meaningfully average two target values?
    │   ├── Yes (avg of $50k and $70k = $60k makes sense)  → Regression
    │   └── No  (avg of "cat" and "dog" = ??? nonsense)    → Classification
    └── Is there a natural ordering with infinite in-between values?
        ├── Yes (temperature: 20.0, 20.1, 20.15...)        → Regression
        └── No  (blood type: A, B, AB, O — no "in between") → Classification
```

**Real-world gut check:**
| If you're predicting...                  | It's...            |
|------------------------------------------|--------------------|
| House price in dollars                   | Regression         |
| Whether a customer will churn (yes/no)   | Classification     |
| Tomorrow's temperature in °C             | Regression         |
| Type of disease (Type-A / Type-B / None) | Classification     |
| A student's exam score (0–100)           | Regression         |
| Whether an email is spam or not          | Classification     |
| Stock price next week                    | Regression         |
| Sentiment (positive/negative/neutral)    | Classification     |
| Number of units sold                     | Regression         |
| Whether a loan will default              | Classification     |

---

## 📈 Regression Decision Tree

Follow the questions top-to-bottom. Each leaf node (marked with ✅) is your recommended starting algorithm.

```
START: You have a Regression problem
│
├── Q1: How many features do you have?
│   │
│   ├── Few features (1–15) and you suspect LINEAR relationships?
│   │   │
│   │   ├── Q2: Is multicollinearity a concern?
│   │   │   │   (Are your features correlated with each other?)
│   │   │   │
│   │   │   ├── No → Q3: Do you need a simple, interpretable baseline?
│   │   │   │   ├── Yes
│   │   │   │   │   ✅ LINEAR REGRESSION (OLS)
│   │   │   │   │   Example: Predicting salary from years of experience
│   │   │   │   │
│   │   │   │   └── No → Q4: Is the relationship curved/non-linear?
│   │   │   │       ├── Yes
│   │   │   │       │   ✅ POLYNOMIAL REGRESSION
│   │   │   │       │   Example: Bacterial growth rate vs. temperature
│   │   │   │       │           (rises, peaks, then falls — a curve)
│   │   │   │       │
│   │   │   │       └── No (linear, but you want regularization for safety)
│   │   │   │           ✅ RIDGE REGRESSION (L2)
│   │   │   │           Example: Predicting crop yield from 10 weather features
│   │   │   │                    where you want to keep all features but shrink
│   │   │   │                    noisy ones
│   │   │   │
│   │   │   └── Yes (features are correlated with each other)
│   │   │       │
│   │   │       ├── Q5: Do you want automatic feature selection?
│   │   │       │   │   (Should the model drop useless features for you?)
│   │   │       │   │
│   │   │       │   ├── Yes
│   │   │       │   │   ✅ LASSO REGRESSION (L1)
│   │   │       │   │   Example: Predicting hospital readmission from 200
│   │   │       │   │            medical features — Lasso zeros out the
│   │   │       │   │            irrelevant ones automatically
│   │   │       │   │
│   │   │       │   └── Want BOTH feature selection AND handling correlation?
│   │   │       │       ✅ ELASTIC NET (L1 + L2 combined)
│   │   │       │       Example: Genomics — predicting disease severity from
│   │   │       │                thousands of gene expressions where many genes
│   │   │       │                are correlated in groups
│   │   │       │
│   │   │       └── (If unsure, start with Ridge; switch to Lasso/ElasticNet
│   │   │            if you need sparsity)
│   │   │
│   │   └── (Proceed to tree-based models below if linear doesn't fit)
│   │
│   ├── Many features (15–hundreds) OR you suspect NON-LINEAR relationships?
│   │   │
│   │   ├── Q6: Do you need interpretability?
│   │   │   │   (Must you explain WHY the model predicted a certain value?)
│   │   │   │
│   │   │   ├── Yes
│   │   │   │   ✅ DECISION TREE REGRESSOR
│   │   │   │   Example: Explaining to a bank manager why a customer's credit
│   │   │   │            score was predicted as 680 — "because income > $50k
│   │   │   │            AND debt-to-income ratio < 0.3"
│   │   │   │   ⚠️  Prone to overfitting — use max_depth, min_samples_leaf
│   │   │   │
│   │   │   └── No (accuracy matters more than explainability)
│   │   │       │
│   │   │       ├── Q7: How large is your dataset?
│   │   │       │   │
│   │   │       │   ├── Small to Medium (< 50,000 rows)
│   │   │       │   │   │
│   │   │       │   │   ├── Q8: Is training speed a priority?
│   │   │       │   │   │   ├── Yes
│   │   │       │   │   │   │   ✅ RANDOM FOREST REGRESSOR
│   │   │       │   │   │   │   Example: Predicting Airbnb listing prices from
│   │   │       │   │   │   │            location, reviews, amenities — robust
│   │   │       │   │   │   │            out-of-the-box with little tuning
│   │   │       │   │   │   │
│   │   │       │   │   │   └── No (willing to tune for best accuracy)
│   │   │       │   │   │       ✅ SVR (Support Vector Regression)
│   │   │       │   │   │       Example: Predicting stock volatility from a
│   │   │       │   │   │                small, clean financial dataset with
│   │   │       │   │   │                complex non-linear patterns
│   │   │       │   │   │
│   │   │       │   │   └── (KNN Regressor is also an option here — see below)
│   │   │       │   │
│   │   │       │   └── Large (50,000+ rows)
│   │   │       │       │
│   │   │       │       ├── Q9: Do you need top-tier accuracy and can tune?
│   │   │       │       │   ├── Yes
│   │   │       │       │   │   ✅ GRADIENT BOOSTING (XGBoost / LightGBM / CatBoost)
│   │   │       │       │   │   Example: Predicting insurance claim amounts from
│   │   │       │       │   │            customer demographics + claim history
│   │   │       │       │   │            — this is what wins Kaggle competitions
│   │   │       │       │   │
│   │   │       │       │   └── No (want something solid without heavy tuning)
│   │   │       │       │       ✅ RANDOM FOREST REGRESSOR
│   │   │       │       │       Example: Predicting energy consumption of buildings
│   │   │       │       │                from sensor data — reliable and parallelizable
│   │   │       │       │
│   │   │       │       └── Dataset is VERY large (millions) with complex patterns?
│   │   │       │           ✅ NEURAL NETWORK (MLP Regressor)
│   │   │       │           Example: Predicting real-time ride-share pricing from
│   │   │       │                    location, time, demand, weather, events, etc.
│   │   │       │                    — when you have massive data and GPU resources
│   │   │       │
│   │   │       └── (AdaBoost Regressor is a lighter boosting alternative —
│   │   │            see cards below)
│   │   │
│   │   └── Special case: Very few samples (< 500), many features?
│   │       ✅ RIDGE or ELASTIC NET (regularization prevents overfitting)
│   │       Example: Predicting patient outcomes from 500 gene expression
│   │                features but only 100 patients
│   │
│   └── Don't know much about the data? Want a quick, easy baseline?
│       ✅ KNN REGRESSOR
│       Example: Predicting a used car's price based on similar cars sold
│                recently — "find the 5 most similar cars, average their prices"
│       ⚠️  Slow on large datasets, sensitive to feature scaling
```

**Regression Quick-Pick Summary:**

| Situation | Go-To Model |
|-----------|-------------|
| Simple, linear, interpretable | Linear Regression |
| Linear but features are correlated | Ridge / Lasso / Elastic Net |
| Curved relationship, few features | Polynomial Regression |
| Need to explain decisions to humans | Decision Tree Regressor |
| Solid all-rounder, minimal tuning | Random Forest Regressor |
| Maximum accuracy, willing to tune | XGBoost / LightGBM / CatBoost |
| Small dataset, complex patterns | SVR |
| Millions of rows, deep patterns | Neural Network (MLP) |
| Quick lazy baseline | KNN Regressor |

---

## 📊 Classification Decision Tree

```
START: You have a Classification problem
│
├── Q1: How many classes?
│   │
│   ├── Binary (2 classes: yes/no, spam/not-spam, 0/1)
│   │   │
│   │   ├── Q2: Do you need INTERPRETABILITY?
│   │   │   │   (Must you explain decisions to stakeholders, regulators, doctors?)
│   │   │   │
│   │   │   ├── Yes
│   │   │   │   │
│   │   │   │   ├── Q3: Is the relationship between features and outcome
│   │   │   │   │       roughly linear (in log-odds)?
│   │   │   │   │   │
│   │   │   │   │   ├── Yes
│   │   │   │   │   │   ✅ LOGISTIC REGRESSION
│   │   │   │   │   │   Example: Predicting loan default (yes/no) from income,
│   │   │   │   │   │            credit score, debt ratio — banks use this
│   │   │   │   │   │            because regulators demand explainability
│   │   │   │   │   │
│   │   │   │   │   └── No / Not sure
│   │   │   │   │       ✅ DECISION TREE CLASSIFIER
│   │   │   │   │       Example: ER triage — "if heart rate > 120 AND age > 60
│   │   │   │   │                AND chest pain = yes → HIGH PRIORITY"
│   │   │   │   │                Doctors can follow the logic step by step.
│   │   │   │   │       ⚠️  Limit depth to avoid overfitting
│   │   │   │   │
│   │   │   │   └── (Both Logistic Regression and Decision Trees give you
│   │   │   │        feature importances you can show to non-technical people)
│   │   │   │
│   │   │   └── No (accuracy is the priority)
│   │   │       │
│   │   │       ├── Q4: How large is your dataset?
│   │   │       │   │
│   │   │       │   ├── Small (< 1,000 rows)
│   │   │       │   │   │
│   │   │       │   │   ├── Q5: Is it a text or categorical-heavy dataset?
│   │   │       │   │   │   ├── Yes
│   │   │       │   │   │   │   ✅ NAIVE BAYES (Multinomial for text,
│   │   │       │   │   │   │                    Gaussian for continuous,
│   │   │       │   │   │   │                    Bernoulli for binary features)
│   │   │       │   │   │   │   Example: Email spam filtering — works shockingly
│   │   │       │   │   │   │            well even on tiny datasets because it
│   │   │       │   │   │   │            only needs word frequencies
│   │   │       │   │   │   │
│   │   │       │   │   │   └── No
│   │   │       │   │   │       ✅ KNN CLASSIFIER
│   │   │       │   │   │       Example: Classifying iris flowers by petal/sepal
│   │   │       │   │   │                measurements — "this flower looks most
│   │   │       │   │   │                like the 5 nearest Setosa samples"
│   │   │       │   │   │       ⚠️  ALWAYS scale your features first (StandardScaler)
│   │   │       │   │   │
│   │   │       │   │   └── (Logistic Regression is also strong on small data)
│   │   │       │   │
│   │   │       │   ├── Medium (1,000 – 100,000 rows)
│   │   │       │   │   │
│   │   │       │   │   ├── Q6: Is the data linearly separable?
│   │   │       │   │   │   │   (Can you draw a straight line/plane between classes?)
│   │   │       │   │   │   │
│   │   │       │   │   │   ├── Yes (or mostly)
│   │   │       │   │   │   │   ✅ SVM (Linear Kernel)
│   │   │       │   │   │   │   Example: Classifying tumors as malignant/benign
│   │   │       │   │   │   │            from cell measurements — SVM finds the
│   │   │       │   │   │   │            widest possible margin between classes
│   │   │       │   │   │   │
│   │   │       │   │   │   └── No (complex, twisted decision boundaries)
│   │   │       │   │   │       │
│   │   │       │   │   │       ├── Q7: Do you want robustness with little tuning?
│   │   │       │   │   │       │   ├── Yes
│   │   │       │   │   │       │   │   ✅ RANDOM FOREST CLASSIFIER
│   │   │       │   │   │       │   │   Example: Predicting customer churn from
│   │   │       │   │   │       │   │            usage patterns, demographics,
│   │   │       │   │   │       │   │            support tickets — handles mixed
│   │   │       │   │   │       │   │            feature types gracefully
│   │   │       │   │   │       │   │
│   │   │       │   │   │       │   └── No (want maximum accuracy, will tune)
│   │   │       │   │   │       │       ✅ SVM (RBF / Polynomial Kernel)
│   │   │       │   │   │       │       Example: Handwritten digit recognition
│   │   │       │   │   │       │                (MNIST subset) — RBF kernel maps
│   │   │       │   │   │       │                data into higher dimensions where
│   │   │       │   │   │       │                a clean boundary exists
│   │   │       │   │   │       │
│   │   │       │   │   │       └── (Both are excellent here — Random Forest is
│   │   │       │   │   │            easier, SVM-RBF can be more accurate with tuning)
│   │   │       │   │   │
│   │   │       │   │   └── Not sure about separability? → Start with Random Forest
│   │   │       │   │
│   │   │       │   └── Large (100,000+ rows)
│   │   │       │       │
│   │   │       │       ├── Q8: Need the absolute best accuracy?
│   │   │       │       │   ├── Yes
│   │   │       │       │   │   ✅ GRADIENT BOOSTING (XGBoost / LightGBM / CatBoost)
│   │   │       │       │   │   Example: Fraud detection in banking — millions of
│   │   │       │       │   │            transactions, need to catch 0.1% fraud
│   │   │       │       │   │            with minimal false positives. Gradient
│   │   │       │       │   │            boosting handles class imbalance well
│   │   │       │       │   │            with scale_pos_weight parameter.
│   │   │       │       │   │
│   │   │       │       │   └── No (want fast training + good accuracy)
│   │   │       │       │       ✅ RANDOM FOREST CLASSIFIER
│   │   │       │       │       Example: Network intrusion detection — classifying
│   │   │       │       │                traffic as normal/attack from packet features.
│   │   │       │       │                Trains in parallel on all CPU cores.
│   │   │       │       │
│   │   │       │       └── Massive (millions+) with deep non-linear patterns?
│   │   │       │           ✅ NEURAL NETWORK (MLP Classifier)
│   │   │       │           Example: Image-based medical diagnosis, voice
│   │   │       │                    recognition, NLP tasks — when you have
│   │   │       │                    the data AND the compute to justify it
│   │   │       │
│   │   │       └── (AdaBoost is a simpler boosting option — see cards below)
│   │   │
│   │   └── Special case: Extremely imbalanced data (99% vs 1%)?
│   │       → Use SMOTE / class_weight='balanced' with ANY of the above
│   │       → Gradient Boosting and Random Forest handle imbalance best
│   │       → Evaluate with F1-score or AUC-ROC, NOT accuracy
│   │
│   └── Multi-class (3+ classes: cat/dog/bird, digit 0-9, disease types)
│       │
│       ├── Q9: Is it a text classification problem?
│       │   ├── Yes
│       │   │   ✅ NAIVE BAYES (Multinomial) as baseline
│       │   │   ✅ LOGISTIC REGRESSION (with one-vs-rest) as strong second
│       │   │   Example: Classifying news articles into Sports / Politics /
│       │   │            Tech / Entertainment from word frequencies
│       │   │
│       │   └── No (structured/tabular data)
│       │       │
│       │       ├── Q10: Few classes (3–10)?
│       │       │   │
│       │       │   ├── Need interpretability?
│       │       │   │   ├── Yes → ✅ DECISION TREE or LOGISTIC REGRESSION (OvR)
│       │       │   │   └── No  → ✅ RANDOM FOREST or GRADIENT BOOSTING
│       │       │   │
│       │       │   └── Example: Classifying wine quality into Low/Medium/High
│       │       │         from chemical properties
│       │       │
│       │       └── Many classes (10–1000+)?
│       │           ✅ GRADIENT BOOSTING (natively handles multi-class)
│       │           ✅ NEURAL NETWORK (softmax output layer)
│       │           Example: Classifying 100 species of plants from leaf
│       │                    measurements — need a model that scales to many
│       │                    output classes efficiently
│       │
│       └── Special case: Ordinal classes (low < medium < high)?
│           → Treat as regression OR use ordinal encoding + any classifier
│           → Logistic Regression with ordinal encoding works well here
```

**Classification Quick-Pick Summary:**

| Situation | Go-To Model |
|-----------|-------------|
| Need to explain to regulators/doctors | Logistic Regression or Decision Tree |
| Text data (emails, reviews, articles) | Naive Bayes → Logistic Regression |
| Small dataset, quick baseline | KNN or Naive Bayes |
| Medium data, don't know much about it | Random Forest (safe default) |
| Need best accuracy, will tune | XGBoost / LightGBM / CatBoost |
| Linearly separable, medium data | SVM (Linear) |
| Complex boundaries, medium data | SVM (RBF) or Random Forest |
| Millions of rows, deep patterns | Neural Network (MLP) |
| Highly imbalanced classes | Gradient Boosting + SMOTE |

---

## 🃏 Algorithm Deep-Dive Cards

Each card contains: what it does, a memorable real-world example, when to use it,
when NOT to use it, and key hyperparameters to tune.

---

### Regression Algorithms

---

#### 1. Linear Regression (OLS — Ordinary Least Squares)

**What it does:** Finds the best straight line (or flat plane in multiple dimensions)
that minimizes the sum of squared errors between predictions and actual values.

**Memorable Example:**
> Predicting a house's price from its square footage. Plot the data, draw the best-fit
> line through the dots — that's Linear Regression. "Every extra 100 sq ft adds ~$15,000."

**When to use:**
- Linear relationship between features and target
- You need a fast, interpretable baseline
- Feature coefficients must be explainable ("feature X increases price by $Y")
- Few features, enough samples (n > features)

**When NOT to use:**
- Relationship is curved (use Polynomial or tree-based models)
- Features are highly correlated (multicollinearity inflates coefficients — use Ridge/Lasso)
- Lots of outliers (OLS is sensitive to outliers — consider Huber regression)
- High-dimensional data with more features than samples (will overfit or fail)

**Key Hyperparameters:** None for basic OLS (that's the beauty — and the limitation).

**Metrics to watch:** R², Adjusted R², MAE, RMSE, residual plots for pattern detection.

---

#### 2. Polynomial Regression

**What it does:** Extends Linear Regression by adding squared, cubed (etc.) versions of
features so the model can fit curves instead of straight lines.

**Memorable Example:**
> Predicting fuel efficiency (MPG) from engine horsepower. Low HP → decent MPG,
> medium HP → best MPG (sweet spot), high HP → terrible MPG. That U-shape needs a curve,
> not a straight line. A degree-2 polynomial captures this perfectly.

**When to use:**
- Clear curved/non-linear pattern in scatter plots
- Few features (polynomial expansion creates n^degree features — explodes fast)
- You've tried Linear Regression and the residuals show a curved pattern

**When NOT to use:**
- Many features (degree 3 on 20 features = thousands of new columns → overfitting)
- High polynomial degree without validation (degree > 4 almost always overfits)
- When tree-based models would capture non-linearity more naturally

**Key Hyperparameters:**
- `degree` (start with 2, rarely go above 4, validate with cross-validation)

**Pro tip:** Always use with Ridge/Lasso regularization to prevent overfitting at higher degrees.

---

#### 3. Ridge Regression (L2 Regularization)

**What it does:** Linear Regression + a penalty that shrinks all coefficients toward zero
(but never exactly to zero). Controlled by the parameter alpha (α).

**Memorable Example:**
> Predicting crop yield from 10 weather features (temperature, humidity, rainfall, wind,
> etc.) that are all somewhat correlated. Ridge says: "I'll use ALL features, but I'll
> turn down the volume on the noisy ones so no single feature dominates."

**When to use:**
- Multicollinearity is present (correlated features)
- You want to keep all features but prevent overfitting
- Slightly better than OLS when you have more features than ideal

**When NOT to use:**
- You need true feature selection (Ridge keeps all features, just shrinks them)
- The dataset is perfectly clean with no multicollinearity (OLS is simpler)

**Key Hyperparameters:**
- `alpha` — higher = more regularization, more shrinkage (use `RidgeCV` to auto-tune)

---

#### 4. Lasso Regression (L1 Regularization)

**What it does:** Like Ridge, but the penalty can shrink coefficients all the way to ZERO,
effectively removing features from the model. Built-in feature selection.

**Memorable Example:**
> A hospital has 200 blood test measurements per patient and wants to predict readmission
> risk. Lasso says: "Actually, only 12 of these measurements matter. I'll zero out the
> other 188." Now the model is simpler, faster, and easier to explain.

**When to use:**
- You suspect many features are irrelevant or redundant
- You want automatic feature selection
- Interpretability matters (fewer non-zero coefficients = simpler story)

**When NOT to use:**
- Features are correlated in groups (Lasso picks one from each group randomly — use Elastic Net)
- You need ALL features to remain in the model (use Ridge)

**Key Hyperparameters:**
- `alpha` — higher = more features dropped (use `LassoCV` to auto-tune)

---

#### 5. Elastic Net (L1 + L2 Combined)

**What it does:** Combines Ridge and Lasso penalties. Gets the feature selection of Lasso
AND the stability of Ridge when features are correlated.

**Memorable Example:**
> Genomics: predicting disease severity from 10,000 gene expressions. Many genes work in
> groups (Gene A and Gene B always activate together). Lasso would randomly drop one from
> each pair. Elastic Net keeps correlated genes together while still eliminating irrelevant ones.

**When to use:**
- High-dimensional data with correlated feature groups
- You want feature selection (like Lasso) but more stable results
- When Lasso's results change wildly between runs

**When NOT to use:**
- Simple problems where Ridge or Lasso alone works fine (unnecessary complexity)

**Key Hyperparameters:**
- `alpha` — overall regularization strength
- `l1_ratio` — 0 = pure Ridge, 1 = pure Lasso, 0.5 = balanced mix

---

#### 6. Decision Tree Regressor

**What it does:** Splits data into groups using if/else rules, then predicts the average
value in each final group (leaf). Creates a tree of binary decisions.

**Memorable Example:**
> Predicting employee salary: "If years_experience > 5 AND department = Engineering
> AND has_masters = True → average salary = $95,000." A manager can follow this logic
> without any statistics knowledge.

**When to use:**
- Interpretability is critical (explain predictions to non-technical people)
- Data has non-linear relationships and interactions between features
- Mixed feature types (numerical + categorical) with no preprocessing needed
- Quick exploration before building ensemble models

**When NOT to use:**
- You need high accuracy (single trees overfit easily — use Random Forest instead)
- Data has smooth, continuous relationships (trees create blocky step-function predictions)
- Small datasets (very prone to overfitting)

**Key Hyperparameters:**
- `max_depth` (3–10 for interpretability, deeper = more overfitting risk)
- `min_samples_leaf` (increase to prevent tiny leaves)
- `min_samples_split`

---

#### 7. Random Forest Regressor

**What it does:** Trains hundreds of Decision Trees, each on a random subset of data and
features, then averages their predictions. The "wisdom of crowds" for trees.

**Memorable Example:**
> Predicting Airbnb listing prices. One tree might focus on location + bedrooms,
> another on reviews + amenities, another on neighborhood + host rating. Individually
> each tree is mediocre, but averaged together? Surprisingly accurate. It's like asking
> 100 real estate agents and averaging their estimates — better than any single agent.

**When to use:**
- You want a strong model with minimal tuning (excellent "out-of-the-box")
- Mixed feature types, missing values, outliers — it handles everything
- You need feature importance rankings
- Parallel training (scales well on multi-core CPUs)

**When NOT to use:**
- Real-time low-latency prediction (hundreds of trees = slower inference)
- You need the absolute last 1% of accuracy (Gradient Boosting usually wins)
- Very high-dimensional sparse data (text/NLP — use specialized models)
- Memory-constrained environments (stores hundreds of full trees)

**Key Hyperparameters:**
- `n_estimators` (100–500 trees; more = better but diminishing returns)
- `max_depth` (None for full depth, or limit for speed)
- `max_features` ('sqrt' for classification, 'log2' or 0.33 for regression)
- `min_samples_leaf`

---

#### 8. Gradient Boosting Regressor (XGBoost / LightGBM / CatBoost)

**What it does:** Trains trees SEQUENTIALLY — each new tree specifically fixes the mistakes
of the previous trees. Instead of wisdom of crowds, it's like a team of specialists where
each one focuses on the cases the last one got wrong.

**Memorable Example:**
> Predicting insurance claim amounts. The first tree gets a rough estimate. The second tree
> looks at where the first was off and corrects those errors. The third corrects the
> remaining errors. After 500 iterations, you have a model that wins Kaggle competitions.
> This is the #1 algorithm for structured/tabular data in competitive ML.

**When to use:**
- Structured/tabular data where you need maximum predictive performance
- Kaggle competitions, production ML systems, any situation where accuracy matters most
- Medium to large datasets (10,000+ rows)
- Data with complex non-linear relationships and feature interactions

**When NOT to use:**
- Tiny datasets (< 500 rows) — will overfit even with regularization
- You need real-time training/updates (boosting is sequential → can't parallelize training)
- Interpretability is more important than accuracy
- Quick prototype / baseline (start with Random Forest, upgrade to boosting if needed)

**Key Hyperparameters:**
- `n_estimators` (100–1000; use early stopping to find the right number)
- `learning_rate` (0.01–0.3; lower = more trees needed but better generalization)
- `max_depth` (3–8; shallower than Random Forest since errors compound)
- `subsample` (0.7–0.9; random fraction of rows per tree — prevents overfitting)
- `colsample_bytree` (0.7–0.9; random fraction of features per tree)

**XGBoost vs LightGBM vs CatBoost:**
| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Speed | Fast | Fastest | Medium |
| Categorical handling | Manual encoding | Some support | Best (native) |
| GPU support | Yes | Yes | Yes |
| Default performance | Great | Great | Great with less tuning |
| Best for | General use | Large datasets | Categorical-heavy data |

---

#### 9. SVR (Support Vector Regression)

**What it does:** Finds a "tube" around the best-fit line/curve. Points inside the tube
are considered "close enough" (no penalty). Only points outside the tube contribute to the
model — these are the support vectors.

**Memorable Example:**
> Predicting a chemical reaction's yield from temperature and pressure. The data is noisy
> but the underlying relationship is smooth. SVR ignores small fluctuations (inside the tube)
> and only learns from the important deviations — like a teacher who ignores small mistakes
> and only corrects the big misunderstandings.

**When to use:**
- Small to medium datasets (< 10,000 rows) — does NOT scale well
- Complex non-linear relationships (with RBF kernel)
- You need robustness to outliers (the epsilon-tube ignores them)

**When NOT to use:**
- Large datasets (training time is O(n²) to O(n³) — painfully slow beyond 10K rows)
- You need feature importances (SVR is a black box)
- High-dimensional sparse data (use linear models instead)

**Key Hyperparameters:**
- `kernel` ('rbf' for non-linear, 'linear' for linear)
- `C` (regularization: higher = less tolerant of errors)
- `epsilon` (width of the "no-penalty" tube)
- `gamma` (for RBF kernel: how much influence each point has)

---

#### 10. KNN Regressor (K-Nearest Neighbors)

**What it does:** To predict a new point, finds the K closest points in the training data
and averages their values. No training phase — all work happens at prediction time.

**Memorable Example:**
> Pricing a used car: "Find the 5 most similar cars recently sold (same make, similar
> mileage, similar age), average their sale prices." That's KNN. Simple, intuitive,
> and requires zero math to explain.

**When to use:**
- Quick, interpretable baseline ("it predicted $X because similar items sold for $Y")
- Small datasets where patterns are local
- Non-parametric situations (you don't want to assume any distribution shape)

**When NOT to use:**
- Large datasets (prediction requires scanning ALL training points — very slow)
- High-dimensional data ("curse of dimensionality" — distance becomes meaningless in 50+ dims)
- Features are on different scales and you forget to normalize (distance is dominated by
  the feature with the largest range)

**Key Hyperparameters:**
- `n_neighbors` (k: typically 3–15; use cross-validation to find the best)
- `weights` ('uniform' = equal vote, 'distance' = closer neighbors matter more)
- `metric` ('minkowski' default; try 'manhattan' for sparse data)

**Critical:** Always use `StandardScaler` or `MinMaxScaler` before KNN.

---

#### 11. AdaBoost Regressor

**What it does:** Another boosting method, but simpler than Gradient Boosting. Trains weak
learners (usually small Decision Trees) sequentially, re-weighting samples so that
hard-to-predict points get more attention in later rounds.

**Memorable Example:**
> A team of tutors helping a struggling student. The first tutor teaches the basics.
> The second tutor focuses on topics the student still got wrong. The third tutor
> drills on the remaining weak spots. Each tutor "boosts" the overall learning.

**When to use:**
- Want a simple boosting baseline before trying XGBoost
- Small to medium datasets
- Less prone to overfitting than Gradient Boosting on noisy data

**When NOT to use:**
- You need maximum accuracy (Gradient Boosting variants almost always beat AdaBoost)
- Very noisy data with many outliers (AdaBoost keeps upweighting outliers → poor results)

**Key Hyperparameters:**
- `n_estimators` (50–200)
- `learning_rate` (0.01–1.0)
- `base_estimator` (default: Decision Tree with max_depth=3)

---

#### 12. Neural Network — MLP Regressor (Multi-Layer Perceptron)

**What it does:** Layers of interconnected "neurons" that learn complex non-linear
mappings from inputs to outputs through backpropagation. Universal function approximators.

**Memorable Example:**
> Predicting real-time Uber surge pricing. Input: location coordinates, time of day,
> day of week, local events, weather, driver supply, rider demand. The relationships
> between these features are incredibly complex and intertwined. A neural network
> can learn patterns that no linear model or even tree model would capture.

**When to use:**
- Very large datasets (100K+ rows) where you have compute resources
- Highly complex, non-linear relationships between features
- When all simpler models have plateaued in performance
- Unstructured data inputs (images, text, audio) combined with tabular data

**When NOT to use:**
- Small datasets (will overfit catastrophically)
- Interpretability is required (neural networks are the ultimate black box)
- You don't have GPU access or time for hyperparameter tuning
- Simpler models haven't been tried yet (always start simple)
- Tabular data competitions (Gradient Boosting almost always wins on tabular data)

**Key Hyperparameters:**
- `hidden_layer_sizes` (e.g., (100, 50) = two layers with 100 and 50 neurons)
- `activation` ('relu' is the safe default)
- `learning_rate_init` (0.001 is standard)
- `batch_size`, `max_iter`, `early_stopping`

---

### Classification Algorithms

---

#### 13. Logistic Regression

**What it does:** Despite the name, this is a CLASSIFICATION algorithm. It fits a linear
boundary and outputs probabilities (0% to 100%) using the sigmoid function.

**Memorable Example:**
> A bank predicting loan default. The model outputs: "This applicant has a 73% probability
> of defaulting." The bank sets a threshold (e.g., 50%) to decide: approve or reject.
> Regulators can inspect the coefficients: "Each $10,000 of debt increases default
> probability by 4%." Complete transparency.

**When to use:**
- Binary classification where you need probability outputs
- Interpretability and explainability required (regulated industries: banking, healthcare)
- Linear or near-linear decision boundary
- Baseline model before trying anything more complex
- Text classification (with TF-IDF features — surprisingly competitive)

**When NOT to use:**
- Complex non-linear decision boundaries (use SVM-RBF or tree-based models)
- You have very few samples and many features without regularization
- The problem fundamentally requires capturing feature interactions (trees are better)

**Key Hyperparameters:**
- `C` (inverse of regularization strength; lower = more regularization)
- `penalty` ('l1' for feature selection, 'l2' for default, 'elasticnet' for both)
- `solver` ('lbfgs' default; 'saga' for large datasets; 'liblinear' for small)
- `class_weight` ('balanced' for imbalanced datasets)

---

#### 14. Naive Bayes (Gaussian / Multinomial / Bernoulli)

**What it does:** Applies Bayes' theorem with a "naive" assumption that all features are
independent of each other. Despite this often-wrong assumption, it works remarkably well
in practice, especially for text.

**Memorable Example:**
> Spam email filter: "What's the probability this email is spam given it contains the words
> 'free', 'winner', and 'click here'?" Naive Bayes calculates: P(spam | these words) using
> simple word frequency counts. Gmail's original spam filter was Naive Bayes.

**Variants:**
| Variant | Feature Type | Example |
|---------|-------------|---------|
| **GaussianNB** | Continuous (normally distributed) | Sensor readings, measurements |
| **MultinomialNB** | Counts / Frequencies | Word counts in text documents |
| **BernoulliNB** | Binary (0/1) | Word presence/absence in text |

**When to use:**
- Text classification (spam, sentiment, topic categorization)
- Very small training datasets (needs minimal data to estimate probabilities)
- Real-time classification (prediction is extremely fast — just multiplication)
- Multi-class problems with many categories
- As a quick baseline (takes seconds to train)

**When NOT to use:**
- Features are heavily correlated (violates the independence assumption badly)
- You need high accuracy on structured/tabular data (tree models are better)
- Numeric features with complex distributions (Gaussian assumption may fail)

**Key Hyperparameters:**
- `alpha` (Laplace smoothing: prevents zero probabilities for unseen feature values; default=1.0)
- `var_smoothing` (GaussianNB: portion of largest variance added to all variances for stability)

---

#### 15. K-Nearest Neighbors Classifier

**What it does:** Classifies a new point by looking at its K nearest neighbors in the
training data and taking a majority vote. "You are the average of the 5 people closest to you."

**Memorable Example:**
> A new student transfers to school. Which friend group will they join? Look at the 5
> students most similar to them (shared interests, same neighborhood, similar grades).
> If 3 of the 5 are in the Science Club → the new student probably joins Science Club too.

**When to use:**
- Small datasets where local patterns matter
- Multi-class classification with complex decision boundaries
- Baseline model with zero assumptions about data distribution
- When you want to explain predictions intuitively ("classified as X because nearest neighbors are X")

**When NOT to use:**
- Large datasets (O(n) prediction time — scans entire training set for EVERY prediction)
- High-dimensional data (50+ features — distances become meaningless, "curse of dimensionality")
- Features on different scales without normalization (MUST StandardScale first)
- Real-time applications with latency requirements
- Data with many irrelevant features (all features affect distance equally)

**Key Hyperparameters:**
- `n_neighbors` (k: odd numbers avoid ties in binary classification; try 3, 5, 7, 11)
- `weights` ('uniform' or 'distance')
- `metric` ('minkowski', 'euclidean', 'manhattan')

---

#### 16. Support Vector Machine (SVM) Classifier

**What it does:** Finds the hyperplane (decision boundary) that maximizes the margin
(distance) between the two closest points of different classes. With kernels, it can
create non-linear boundaries by projecting data into higher dimensions.

**Memorable Example:**
> Imagine red and blue marbles on a table. SVM draws the line that's as far as possible
> from both the nearest red AND the nearest blue marble. This maximum-margin line
> generalizes best to new marbles. If the marbles are mixed together (not linearly separable),
> the RBF kernel "lifts" them into 3D space where they CAN be separated by a flat plane.

**When to use:**
- Medium datasets (1,000–50,000 rows) — sweet spot for SVM
- Clear margin of separation between classes
- High-dimensional feature spaces (text with TF-IDF, genomics)
- Binary classification with well-defined boundaries
- When you want strong theoretical guarantees (maximum margin theory)

**When NOT to use:**
- Large datasets (> 100,000 rows) — training is O(n²) to O(n³), extremely slow
- You need probability outputs (SVM doesn't natively produce probabilities;
  `probability=True` is slow and uses Platt scaling as an approximation)
- You need feature importances (SVM is a black box)
- Noisy data with overlapping classes (SVM tries hard to separate them → overfitting)

**Key Hyperparameters:**
- `kernel` ('linear', 'rbf', 'poly')
- `C` (regularization: low = wider margin + more misclassifications, high = tighter fit)
- `gamma` (RBF: 'scale' default; higher = more complex boundary; lower = smoother)

---

#### 17. Decision Tree Classifier

**What it does:** Builds a tree of yes/no questions that splits the data until each leaf
is (mostly) one class. The tree can be printed and followed by a human.

**Memorable Example:**
> Hospital ER triage decision tree:
> "Chest pain? → Yes → Age > 50? → Yes → Heart rate > 100? → Yes → CRITICAL PRIORITY"
> A nurse can follow this tree without any statistical knowledge.
> This is why decision trees are used in medicine, law, and finance where decisions
> must be explainable and auditable.

**When to use:**
- Interpretability is the #1 requirement
- Need to present the model to non-technical stakeholders
- Mix of categorical and numerical features (no encoding needed)
- Feature interactions are important (trees capture them naturally)
- Quick data exploration before building ensemble models

**When NOT to use:**
- You need high accuracy on its own (single trees overfit — use Random Forest/Boosting)
- Smooth, continuous relationships (trees create blocky step-function boundaries)
- Target is sensitive to small changes in data (trees are notoriously unstable —
  remove one data point and the entire tree can change)

**Key Hyperparameters:**
- `max_depth` (3–7 for interpretable trees)
- `min_samples_split`, `min_samples_leaf`
- `criterion` ('gini' or 'entropy' — usually similar performance)
- `class_weight` ('balanced' for imbalanced data)

---

#### 18. Random Forest Classifier

**What it does:** Ensembles hundreds of Decision Trees, each trained on random subsets
of data and features. Final prediction is by majority vote.

**Memorable Example:**
> Predicting whether a patient has diabetes from health metrics. Each tree sees a
> random subset: Tree 1 uses age + BMI + glucose; Tree 2 uses blood pressure + insulin +
> skin thickness; Tree 3 uses age + insulin + BMI. Each tree is a mediocre predictor,
> but their combined vote is highly accurate. This is "ensemble wisdom" — the same
> reason a panel of judges is better than one judge.

**When to use:**
- The "Swiss Army Knife" of ML — works well on almost everything
- You want strong performance with minimal tuning
- Mixed feature types, outliers, missing values (handles all gracefully)
- You need feature importance rankings
- First model to try when you don't know what to use

**When NOT to use:**
- Latency-sensitive real-time prediction (hundreds of trees = slower)
- You need the absolute best accuracy (Gradient Boosting usually wins by 1–3%)
- Extremely high-dimensional sparse data (e.g., text with 50,000 TF-IDF features)
- Memory-constrained deployment (each tree is stored in memory)

**Key Hyperparameters:**
- `n_estimators` (100–500; more is generally better but with diminishing returns)
- `max_depth` (None = fully grown; limit for speed or to reduce overfitting)
- `max_features` ('sqrt' is the standard default for classification)
- `class_weight` ('balanced' or 'balanced_subsample' for imbalanced classes)

---

#### 19. Gradient Boosting Classifier (XGBoost / LightGBM / CatBoost)

**What it does:** Sequentially trains small trees, where each new tree focuses on correcting
the mistakes of all previous trees combined. The "master class" of tabular ML.

**Memorable Example:**
> Credit card fraud detection: 99.9% of transactions are legitimate, 0.1% are fraud.
> The first tree makes a rough classifier. The second tree focuses on the cases the first
> got wrong (missed frauds and false alarms). By tree #300, the model catches 95% of
> fraud with only 0.5% false positives. This is the algorithm behind most real-world
> fraud detection, ad click prediction, and recommendation systems.

**When to use:**
- Maximum accuracy on structured/tabular data (the king of Kaggle)
- Medium to large datasets (10,000+ rows)
- Class imbalance (use `scale_pos_weight` or `class_weight`)
- You have time and resources for hyperparameter tuning
- Production ML systems where every 0.1% accuracy matters

**When NOT to use:**
- Tiny datasets (< 500 rows) — will overfit
- Need for real-time model retraining (sequential nature = slower training than RF)
- Interpretability is paramount (use Decision Tree or Logistic Regression)
- Unstructured data like raw images/text (use deep learning instead)

**Key Hyperparameters:**
- `n_estimators` + `early_stopping_rounds` (let it auto-determine the right number)
- `learning_rate` (0.01–0.1 for best generalization; lower = more trees needed)
- `max_depth` (3–6 is typical; shallower than RF since trees are additive)
- `subsample`, `colsample_bytree` (0.7–0.9 for randomness/regularization)
- `scale_pos_weight` (set to count(negative)/count(positive) for imbalanced data)

---

#### 20. AdaBoost Classifier

**What it does:** Iteratively trains weak classifiers (small stumps/trees), giving more weight
to misclassified samples in each round. Final prediction is a weighted vote of all weak learners.

**Memorable Example:**
> A quiz show where each round's questions get harder. Round 1: easy questions everyone
> gets right. Round 2: focuses on what Round 1 got wrong. Round 3: focuses on what's STILL
> wrong. Each round is a simple classifier, but the combined score across all rounds
> is quite accurate.

**When to use:**
- Quick boosting baseline
- Less prone to overfitting on clean data compared to Gradient Boosting
- Good on small to medium datasets
- Want a simpler implementation than full Gradient Boosting

**When NOT to use:**
- Noisy data with outliers (AdaBoost keeps upweighting misclassified outliers)
- You need best-in-class accuracy (XGBoost/LightGBM outperform it)
- Large-scale datasets (slower and less effective than modern boosting)

**Key Hyperparameters:**
- `n_estimators` (50–300)
- `learning_rate` (0.01–1.0; lower = more conservative)

---

#### 21. Neural Network — MLP Classifier (Multi-Layer Perceptron)

**What it does:** Fully connected layers of neurons that learn complex decision boundaries
through backpropagation. Can approximate any classification function given enough data.

**Memorable Example:**
> Handwritten digit recognition (MNIST): Each image is 28×28 = 784 pixels fed into the
> network. First layers learn edges, middle layers learn curves and corners, final layers
> recognize complete digits. The same principle powers face recognition, voice assistants,
> and self-driving car object detection.

**When to use:**
- Massive datasets (100K+ rows) with complex patterns
- When simpler models have plateaued
- Multi-modal inputs (combining tabular + image + text)
- Problems where feature engineering is hard (the network learns features automatically)

**When NOT to use:**
- Small datasets (< 5,000 rows) — will overfit badly
- Tabular data where Gradient Boosting hasn't been tried (GB usually wins on tabular)
- Interpretability required
- Limited compute / no GPU
- You can't afford extensive hyperparameter tuning

**Key Hyperparameters:**
- `hidden_layer_sizes` (start with (100,), try (100, 50), (256, 128, 64))
- `activation` ('relu' default; try 'tanh' if relu underperforms)
- `solver` ('adam' default; 'lbfgs' for small datasets)
- `learning_rate_init` (0.001), `batch_size`, `early_stopping=True`

---

## 📋 Universal Quick-Reference Table

| # | Algorithm | Type | Best Dataset Size | Training Speed | Prediction Speed | Interpretability | Handles Non-Linearity | Needs Scaling? |
|---|-----------|------|-------------------|----------------|------------------|------------------|-----------------------|----------------|
| 1 | Linear Regression | Reg | Any | ⚡ Very Fast | ⚡ Very Fast | ✅ High | ❌ No | No |
| 2 | Polynomial Regression | Reg | Small-Med | ⚡ Fast | ⚡ Fast | ✅ Medium | ✅ Yes (curves) | No |
| 3 | Ridge Regression | Reg | Any | ⚡ Very Fast | ⚡ Very Fast | ✅ High | ❌ No | Yes |
| 4 | Lasso Regression | Reg | Any | ⚡ Very Fast | ⚡ Very Fast | ✅ High | ❌ No | Yes |
| 5 | Elastic Net | Reg | Any | ⚡ Fast | ⚡ Fast | ✅ High | ❌ No | Yes |
| 6 | Decision Tree | Both | Small-Med | ⚡ Fast | ⚡ Very Fast | ✅ Very High | ✅ Yes | No |
| 7 | Random Forest | Both | Med-Large | 🟡 Medium | 🟡 Medium | 🟡 Medium | ✅ Yes | No |
| 8 | Gradient Boosting | Both | Med-Large | 🔴 Slow | 🟡 Medium | 🟡 Low-Med | ✅ Yes | No |
| 9 | AdaBoost | Both | Small-Med | 🟡 Medium | 🟡 Medium | 🟡 Medium | ✅ Yes | No |
| 10 | SVR / SVM | Both | Small-Med | 🔴 Slow (large n) | 🟡 Medium | ❌ Low | ✅ Yes (kernel) | Yes |
| 11 | KNN | Both | Small | ⚡ None* | 🔴 Very Slow | ✅ High | ✅ Yes | Yes (critical!) |
| 12 | Naive Bayes | Class | Any | ⚡ Very Fast | ⚡ Very Fast | ✅ High | ❌ Limited | Depends† |
| 13 | Logistic Regression | Class | Any | ⚡ Very Fast | ⚡ Very Fast | ✅ High | ❌ No | Yes |
| 14 | Neural Network (MLP) | Both | Large-Huge | 🔴 Very Slow | ⚡ Fast | ❌ Very Low | ✅ Yes | Yes |

> \* KNN has no training phase — it stores all data and computes at prediction time.
>
> † MultinomialNB does not need scaling; GaussianNB assumes normal distribution.

---

## ⚠️ Common Mistakes to Avoid

### 1. Jumping to Complex Models
> **Wrong:** "Let me start with XGBoost and a neural network."
> **Right:** Always start with a simple baseline (Linear/Logistic Regression). If it gets
> 92% accuracy, maybe you don't NEED a 500-tree ensemble. Simple models are faster to train,
> easier to debug, easier to deploy, and easier to explain.

### 2. Forgetting to Scale Features for Distance-Based Models
> KNN, SVM, and Neural Networks use distance calculations. If "age" ranges 0–80 and
> "salary" ranges 20,000–200,000, salary will dominate all distances.
> **Always** use `StandardScaler` or `MinMaxScaler` before these models.
> Tree-based models (Decision Tree, Random Forest, Gradient Boosting) do NOT need scaling.

### 3. Using Accuracy on Imbalanced Data
> If 99% of transactions are legitimate, a model that ALWAYS predicts "legitimate" gets
> 99% accuracy — and catches zero fraud. Use **F1-score**, **Precision-Recall AUC**,
> or **ROC-AUC** instead. Apply `class_weight='balanced'` or SMOTE for imbalanced classes.

### 4. Not Splitting Data Properly
> **Always** split into Train / Validation / Test BEFORE any preprocessing.
> - **Train** (70–75%): Model learns from this.
> - **Validation** (10–15%): Tune hyperparameters using this.
> - **Test** (15–20%): Touch ONCE at the very end for final evaluation.
> Never let test data leak into training or validation.

### 5. Overfitting with Decision Trees
> An unconstrained Decision Tree will memorize the training set (100% training accuracy)
> and fail on new data. **Always** set `max_depth`, `min_samples_leaf`, or use
> Random Forest / Gradient Boosting which have built-in regularization.

### 6. Using SVM or KNN on Large Datasets
> SVM training scales O(n²) to O(n³). KNN prediction scales O(n) per query.
> At 1 million rows, SVM training may take hours/days; KNN prediction will be unbearably slow.
> Switch to Random Forest or Gradient Boosting for large datasets.

### 7. Ignoring the Bias-Variance Tradeoff
> - **High bias** (underfitting): Model is too simple → try more complex models or add features
> - **High variance** (overfitting): Model memorizes training data → add regularization,
>   get more data, or simplify the model
> - **Sweet spot:** Validate with cross-validation, plot learning curves

### 8. Not Looking at Your Data First
> Spending 5 minutes on EDA (scatter plots, histograms, correlation matrix) can save
> 5 hours of debugging a poorly chosen model. Check for: outliers, missing values,
> class imbalance, feature distributions, and relationships between features and target.

---

## 🗺️ The 60-Second Decision Shortcut

If you're short on time, use this ultra-simplified guide:

```
What's your problem?
│
├── REGRESSION
│   ├── Start simple              → Linear Regression
│   ├── Need regularization       → Ridge (keep all) / Lasso (drop some)
│   ├── Want strong default       → Random Forest
│   ├── Want best accuracy        → XGBoost / LightGBM
│   └── Massive data + GPU        → Neural Network
│
└── CLASSIFICATION
    ├── Need explainability       → Logistic Regression / Decision Tree
    ├── Text data                 → Naive Bayes → Logistic Regression
    ├── Small data                → KNN / Naive Bayes
    ├── Want strong default       → Random Forest
    ├── Want best accuracy        → XGBoost / LightGBM
    └── Massive data + GPU        → Neural Network
```

---

*Created as a graduate-level reference for Supervised Learning model selection.*
*Always validate model choice with cross-validation on YOUR specific data — no flowchart replaces empirical testing.*
