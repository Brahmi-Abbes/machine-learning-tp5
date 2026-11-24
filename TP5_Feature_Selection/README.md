# Machine Learning — Practical Session 5: Feature Selection
**National Higher School of Advanced Technologies (ENSTA)**  
**Academic Year:** 2026–2027 | **Level:** First-year Master's Degree  
**Course:** Machine Learning  
**Assignment:** Practical Session 5 — Feature Selection: Filter and Wrapper Methods  

---

## 1. Objectives & Overview

The primary objective of Practical Session 5 is to study, implement, and rigorously benchmark **feature selection** techniques:
- **Filter Methods:** Feature scoring independent of any downstream predictive model (Variance Threshold, Pairwise Correlation Filtering, ANOVA F-test, and Mutual Information).
- **Wrapper Methods:** Subset search driven by model feedback (Forward Feature Selection, Backward Elimination, Recursive Feature Elimination [RFE], and Cross-Validated RFE [RFECV]).
- **Methodological Integrity & Anti-Leakage:** Preventing data leakage by encapsulating scalers, selectors, and estimators within `scikit-learn` `Pipeline` architectures.
- **Robustness Under Noise:** Injecting 20 standard-normal Gaussian noise features to test which feature selection paradigms are fooled by spurious correlations.
- **Empirical Generalization Comparison:** Measuring out-of-sample test accuracy, $F_1$-score, and computational latency on an untouched holdout test partition (171 patients).

---

## 2. Dataset Information

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** Loaded via `sklearn.datasets.load_breast_cancer(as_frame=True)` (original UCI ML Repository: `breast+cancer+wisconsin+diagnostic`)
- **Samples:** 569 observations
- **Target:** Binary classification — `0 = malignant` (212 cases, 37.26%), `1 = benign` (357 cases, 62.74%)
- **Features:** 30 continuous real-valued morphological measurements extracted from digitized fine needle aspirate (FNA) images of breast masses:
  - 10 mean attributes (`mean radius`, ..., `mean fractal dimension`)
  - 10 standard error attributes (`radius error`, ..., `fractal dimension error`)
  - 10 "worst" attributes (`worst radius`, ..., `worst fractal dimension` — mean of the three largest values)
- **Noise Features (Part D):** 20 synthetic noise features (`noise_00` to `noise_19`) generated as independent standard normal draws: $\mathcal{N}(0, 1)$ with `np.random.default_rng(42)`.

---

## 3. Environment & Requirements

The project requires Python 3.9+ and the standard scientific Python stack.

### Required Dependencies (`requirements.txt`)
```txt
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
```

### Installation
From within the `TP5_Feature_Selection/` directory:
```bash
pip install -r requirements.txt
```

### Execution
Run the complete practical session script:
```bash
python tp5.py
```

The script runs deterministically (`random_state=42`), processes all parts sequentially, generates all diagnostic figures, outputs all markdown/console tables, and saves tabular summaries to `results/`.

---

## 4. Repository Structure

```text
machine-learning-tp5/
├── Work_session_5_Feature_selection.pdf   <- Official professor assignment document
└── TP5_Feature_Selection/
    ├── tp5.py                            <- Complete standalone executable script
    ├── requirements.txt                  <- Dependency specification
    ├── README.md                         <- Comprehensive technical report & discussion
    └── results/
        ├── final_comparison_results.csv  <- Final test performance across all 8 methods
        ├── feature_presence_matrix.csv   <- Selected features per method
        └── plots/
            ├── part_a_target_distribution.png
            ├── part_a_correlation_heatmap.png
            ├── part_b_anova_f_scores.png
            ├── part_b_mutual_info_scores.png
            ├── part_b_selectkbest_accuracy_vs_k.png
            ├── part_c_forward_selection.png
            ├── part_c_backward_elimination.png
            ├── part_c_rfecv_accuracy_vs_features.png
            └── part_d_feature_presence_matrix.png
```

---

## 5. Experimental Protocols & Ground Rules

1. **Deterministic Reproducibility:** `random_state=42` is fixed across data generation, partitioning, and cross-validation.
2. **Single Stratified Split:** The dataset is split once into **70% training** ($N_{train} = 398$) and **30% testing** ($N_{test} = 171$) using `train_test_split(..., stratify=y, test_size=0.30, random_state=42)`.
3. **Cross-Validation Protocol:** Internal validation uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the training set only.
4. **Leakage Prevention:** All scalers (`StandardScaler`) and selectors (`SelectKBest`, `SequentialFeatureSelector`, `RFE`) are strictly fitted inside validation folds using `Pipeline`.
5. **Hold-out Test Partition:** The test set is reserved exclusively for the final evaluation in Part D.3.

---

## 6. Detailed Results & Findings

### Part A: Data Inspection and Baseline
- **Dataset Exploration:** 569 rows, 30 continuous features, 0 missing values, 0 duplicated rows.
- **Target Distribution:**
  - Benign (`1`): 357 patients (62.74%)
  - Malignant (`0`): 212 patients (37.26%)
- **Baseline Dummy Classifier:** `DummyClassifier(strategy="most_frequent")` achieves a baseline test accuracy of **62.57%** (predicting the majority benign class).
- **Baseline Logistic Regression Performance (All 30 Features):**
  - Training Accuracy: **98.74%** | Training $F_1$: **0.9901**
  - Testing Accuracy: **98.83%** | Testing $F_1$: **0.9907**
  - Training Execution Time: **0.0086 seconds**
- **Correlation Structure:**
  - *Size geometry redundancy:* `mean radius`, `mean perimeter`, and `mean area` share pairwise correlations $r > 0.98$ ($P \propto r$ and $A \propto r^2$).
  - *Shape irregularity redundancy:* `compactness`, `concavity`, and `concave points` correlate strongly ($r > 0.85$).
  - *Mean vs. Worst statistics:* `worst radius` vs. `mean radius` exhibits $r > 0.90$.

---

### Part B: Filter Methods

#### B.1 Variance Threshold (Unscaled Data)
| Threshold | Features Retained | Features Removed | Dropped Attributes |
|:---------:|:-----------------:|:----------------:|:-------------------|
| **0.00**  | 30                | 0                | *(None)* |
| **0.01**  | 14                | 16               | `mean smoothness`, `mean compactness`, `mean concavity`, `mean concave points`, `mean symmetry`, `mean fractal dimension`, error features, `worst smoothness`, `worst concave points`, `worst symmetry`, `worst fractal dimension` |
| **0.10**  | 11                | 19               | Same as above + `radius error`, `worst compactness`, `worst concavity` |

- **Post-standardization invalidity:** Standardizing features via `StandardScaler` sets all variances identically to $\sigma^2 = 1.0$. Applying a threshold $< 1.0$ retains all features, while $\ge 1.0$ drops all features, rendering variance thresholding completely meaningless after scaling.
- **Limited utility on raw data:** Features possess arbitrary dimensional units (e.g., cell area $\sim 10^5 \text{ mm}^4$, fractal dimension $\sim 10^{-4}$). The filter blindly drops features with small biological scales regardless of their clinical discriminatory power.

#### B.2 Correlation Filtering (`remove_correlated`)
For each feature pair with $|r| > \text{threshold}$, the feature with lower absolute correlation to the target label is dropped.

| Correlation Threshold | Retained Features | Dropped Features | 5-Fold CV Accuracy (Mean ± Std) | Test Accuracy |
|:---------------------:|:-----------------:|:----------------:|:-------------------------------:|:-------------:|
| **0.95**              | 23                | 7                | 0.9824 ± 0.0189                 | 0.9766        |
| **0.90**              | **20**            | **10**           | **0.9824 ± 0.0189**             | **0.9766**    |
| **0.80**              | 12                | 18               | 0.9422 ± 0.0249                 | 0.9415        |

> **Methodological Note on B.2 & General Rules:**  
> On page 3 of the PDF, Section B.2 explicitly asks for the *testing accuracy* of the resulting model. However, the General Rules on page 2 strictly mandate reserving the test set for Part D.3. To respect both requirements without ambiguity, we report both the 5-fold CV validation score and the holdout test accuracy above. Crucially, we select threshold **0.90** based solely on internal validation accuracy (0.9824), eliminating 10 collinear features without test data snooping.

#### B.3 Univariate Scores and `SelectKBest`
- **Spearman Rank Correlation** between ANOVA $F$-statistic and Mutual Information: **0.9657** ($p = 6.10 \times 10^{-18}$), demonstrating high concordant ranking between linear variance separation and non-linear mutual dependence.
- **SelectKBest Cross-Validation Trajectory:**
  - $k \in \{2, 5, 10, 15, 20, 25, 30\}$
  - **ANOVA F-score:** Peak CV accuracy = 0.9874 (at $k=20$, $\sigma = 0.0139$). Smallest $k$ within 1 standard deviation ($0.9874 - 0.0139 = 0.9735$): **$k=20$**.
  - **Mutual Information:** Peak CV accuracy = 0.9824 (at $k=25$, $\sigma = 0.0189$). Smallest $k$ within 1 standard deviation ($0.9824 - 0.0189 = 0.9635$): **$k=20$**.
- **Redundancy Explanation:** Univariate scores evaluate individual margins $P(Y \mid X_i)$ in isolation. Because `worst concave points`, `worst perimeter`, and `worst radius` each exhibit high individual correlation with malignancy, all receive top scores, causing univariate filters to retain collinear feature clusters.

---

### Part C: Wrapper Methods

#### C.1 Forward Selection
- **Approach 1 (`SequentialFeatureSelector` Forward, $k=10$):** Selected `mean radius`, `mean texture`, `mean perimeter`, `mean compactness`, `texture error`, `smoothness error`, `worst radius`, `worst texture`, `worst perimeter`, `worst smoothness`.
- **Approach 2 (Manual Implementation):** Starting from an empty subset, greedily adds the feature maximizing 5-fold CV accuracy. Early stopping criterion ($\text{gain} < \varepsilon = 0.001$) was reached at step 4 (CV score = 0.9849). Continuing search up to $k=10$ achieved **10/10 (100%) exact overlap** with scikit-learn's selection.

#### C.2 Backward Elimination
- **Approach 1 (`SequentialFeatureSelector` Backward, $k=10$):** Retained `texture error`, `area error`, `smoothness error`, `fractal dimension error`, `worst radius`, `worst texture`, `worst perimeter`, `worst smoothness`, `worst concavity`, `worst symmetry`.
- **Forward vs. Backward Overlap:** 6 out of 10 common features.
- **Why they diverge:** Forward selection begins from an empty set and cannot discard a previously selected feature that becomes redundant. Backward elimination begins with all 30 features, assessing features conditioned on the full feature space. Both are greedy heuristics exploring distinct combinatorial trajectories.

#### C.3 Recursive Feature Elimination (RFE & RFECV)
- **RFE (Logistic Regression, $k=10$):** Retained `mean area`, `mean concave points`, `area error`, `worst radius`, `worst texture`, `worst perimeter`, `worst area`, `worst smoothness`, `worst concave points`, `worst symmetry`.
- **RFECV Optimal Feature Count:** **8 features** (peak CV accuracy: **0.9874**).
- **Estimator Divergence:**
  - Linear models (Logistic Regression vs. Linear SVC) share 8/10 features because both rank features via hyperplane normal vector magnitudes ($|w_i|$).
  - Tree-based models (Random Forest) share fewer features (5/10) because trees evaluate non-linear splits and Gini impurity reductions, capturing hierarchical interactions that linear models miss.

#### C.4 Computational Cost of Wrapper Search
For $p = 30$ features, $k = 10$ selected features, and $CV = 5$ folds:
- **Forward Selection Evaluations:**
  $$\sum_{i=0}^{k-1} (p - i) = k \cdot p - \frac{k(k - 1)}{2} = 10 \times 30 - \frac{10 \times 9}{2} = 255 \text{ candidate subsets}$$
  $$\text{Total Models Trained} = 255 \times 5 = 1,275 \text{ models (executed in } 12.40 \text{ s)}$$
- **Exhaustive Combinatorial Search:**
  $$2^p - 1 = 2^{30} - 1 = 1,073,741,823 \text{ candidate subsets}$$
  $$\text{Total Models Required} = 1,073,741,823 \times 5 = 5,368,709,115 \text{ models}$$
  $$\text{Estimated Runtime} \approx 604.3 \text{ days on identical CPU hardware}$$

---

### Part D: Noise Injection, Leakage & Final Benchmark

#### D.1 Noise Injection (50 Features: 30 Real + 20 Gaussian Noise)
Each method was tasked with selecting exactly $k=10$ features from the 50-feature augmented matrix:
| Method | Original Features | Noise Features Selected | Robustness Assessment |
|:-------|:-----------------:|:-----------------------:|:----------------------|
| **SelectKBest (ANOVA F-score)** | **10** | **0** | **Immune:** Real signals have $F \gg 100$; Gaussian noise has $F \sim 1.0$. |
| **Wrapper: RFE (Logistic Regression)** | **10** | **0** | **Immune:** Model coefficients shrink noise weights near zero. |
| **Wrapper: Forward Selection** | **8** | **2** (`noise_00`, `noise_05`) | **Vulnerable:** Spurious random correlation with CV fold boundaries gave artificial accuracy gains. |
| **Wrapper: Backward Elimination** | **10** | **0** | **Robust:** Noise features dropped in early elimination steps. |

#### D.2 Cross-Validation Data Leakage Comparison
On the 50-feature dataset using `SelectKBest(f_classif, k=10)` and `LogisticRegression`:
- **Incorrect Approach (Pre-selection before CV):** 5-fold CV Accuracy = **0.9523**
- **Correct Approach (Selection inside Pipeline CV):** 5-fold CV Accuracy = **0.9523**
- **Observed Gap:** 0.00% on this dataset.
- **Theoretical Explanation:** In Breast Cancer Wisconsin, the top 10 diagnostic morphological signals are overwhelmingly strong ($p < 10^{-20}$) across all folds, so fold partitions do not alter the selected subset. However, in high-dimensional genomic/microarray settings ($n=50, p=10,000$), pre-selection leaks label information, yielding 100% apparent CV accuracy on pure noise where true generalization is 50%.

#### D.3 Final Comparison on the Untouched Test Set (171 Patients)
All pipelines fitted strictly on the 398 training observations and tested once on the 171 holdout test observations:

| Method | Nb features | Test accuracy | Test $F_1$-score | Selection time (s) |
|:-------|:-----------:|:-------------:|:----------------:|:------------------:|
| **Baseline (all features)** | 30 | 0.9883 | 0.9907 | — |
| **Filter: correlation (B.2)** | 20 | 0.9766 | 0.9813 | 0.031 |
| **Filter: SelectKBest, F-score** | 20 | 0.9766 | 0.9813 | 0.009 |
| **Filter: SelectKBest, mutual info** | 20 | 0.9708 | 0.9770 | 0.081 |
| **Wrapper: Forward Selection** | 10 | 0.9357 | 0.9484 | 5.837 |
| **Wrapper: Backward Elimination** | 10 | 0.9591 | 0.9677 | 8.492 |
| **Wrapper: RFE** | 10 | 0.9708 | 0.9767 | 0.116 |
| **Wrapper: RFECV** | **8** | **0.9591** | **0.9671** | **1.134** |

#### Feature Presence Matrix & Consensus Features
- **Consensus Features (Selected by 100% of all 7 selection methods):**
  1. `worst texture` (7/7 methods)
  2. `worst perimeter` (7/7 methods)
- **High-Frequency Predictors (Selected by $\ge 5$ methods):**
  - `worst radius` (6/7)
  - `worst smoothness` (6/7)
  - `area error` (6/7)
  - `worst concave points` (5/7)
  - `worst symmetry` (5/7)
- **Agreement with Part A Correlation Structure:** The consensus selections perfectly match the morphological geometry identified in Part A: instead of keeping all collinear radius/area/perimeter triplets across mean, error, and worst, the algorithms consistently select one dominant extreme boundary measurement (`worst perimeter`), one cell texture measurement (`worst texture`), and standard error indicators (`area error`).

---

## 7. Answers to the Professor's Seven Discussion Questions

### Question 1: Best test accuracy with fewest features & statistical significance
> *Which method achieves the best test accuracy with the fewest features? Is its difference from the baseline statistically meaningful given the test size of 171 patients?*

**Answer:**  
`Filter: correlation (B.2)` and `Filter: SelectKBest, F-score` achieve **97.66% test accuracy** with 20 features (reducing dimensionality by 33%), while `Wrapper: RFE` achieves **97.08% test accuracy** with only 10 features (reducing dimensionality by 67%).  
Statistically, with $N_{test} = 171$, each misclassified patient shifts the test accuracy by $1 / 171 \approx 0.58\%$. The numerical difference between the baseline (98.83%, 2 misclassifications) and RFE (97.08%, 5 misclassifications) is only 3 patients. A two-proportion $z$-test yields $z = 1.15$ ($p = 0.25 > 0.05$). Thus, the difference is **not statistically significant**. Feature selection successfully achieves equivalent diagnostic generalization while eliminating two-thirds of the input variables.

### Question 2: Advantages and limitations of filter vs. wrapper methods
> *What are the advantages and limits of filter methods compared with wrapper methods (speed, redundancy, interaction between features, dependence on the model)?*

**Answer:**  
- **Speed:** Filters are orders of magnitude faster ($O(p)$ computational complexity, executing in milliseconds) compared to wrappers ($O(p \cdot k)$ to $O(p^2)$ cross-validation fits, requiring seconds to minutes).
- **Redundancy:** Univariate filters evaluate features in isolation and frequently select redundant collinear features (e.g., radius, perimeter, and area together). Wrappers assess subset combinations and naturally weed out redundant variables.
- **Interactions:** Filters are blind to feature synergies where individual variables are non-informative alone but predictive in combination. Wrappers evaluate joint predictive power and capture multi-feature interactions.
- **Model Dependence:** Filters are model-agnostic and universally transferable; wrappers tailor their feature subset to the specific inductive bias of the chosen classifier (e.g., linear hyperplanes in Logistic Regression).

### Question 3: Computational expense of wrappers and forward selection formula
> *Why are wrapper methods more expensive? Give the number of models trained by Forward Selection for $p$ features and $k$ selected features.*

**Answer:**  
Wrappers use the machine learning estimator as an evaluation oracle, repeatedly refitting models across cross-validation splits.  
For $p$ initial features and $k$ selected features, Forward Selection evaluates:
$$N_{\text{subsets}} = \sum_{i=0}^{k-1} (p - i) = k \cdot p - \frac{k(k - 1)}{2}$$
With 5-fold cross-validation, the total number of fitted models is $5 \times [k \cdot p - k(k - 1)/2]$. For $p=30$ and $k=10$, this equals $5 \times 255 = \mathbf{1,275}$ models. Backward elimination is even more expensive, requiring $5 \times (30 + 29 + \dots + 11) = \mathbf{2,050}$ models.

### Question 4: Do Forward Selection and Backward Elimination guarantee the optimal subset?
> *Forward Selection and Backward Elimination are greedy. Do they guarantee the optimal subset? Justify using your results.*

**Answer:**  
**No.** Both algorithms are greedy heuristics that commit to myopic local choices without backtracking. Forward selection cannot drop a feature once added if later additions render it obsolete (nesting problem). Backward elimination cannot recover a feature discarded early that would synergize with remaining variables.  
Our empirical results confirm this: Forward Selection and Backward Elimination selected different 10-feature subsets that shared only 6 common features. Finding the globally optimal subset requires exhaustive search over all $2^{30} - 1 = 1,073,741,823$ combinations, which is computationally prohibitive.

### Question 5: Method behavior in the presence of 20 noise features
> *How did the methods behave in front of the 20 noise features, and what does it tell you about each family?*

**Answer:**  
- **SelectKBest (ANOVA F-score)** selected **0 noise features** (100% original features). Because true biological predictors have massive F-statistics ($F > 100$), standard normal noise features ($F \sim 1.0$) could not compete.
- **RFE (Logistic Regression)** also selected **0 noise features**, as linear regularization shrinks noise coefficients near zero during iterative pruning.
- **Forward Selection** was vulnerable, selecting **2 noise features** (`noise_00` and `noise_05`). In greedy forward addition, random empirical alignments of noise features with training fold boundaries produce spurious accuracy spikes, deceiving the search heuristic.

### Question 6: Cross-validation leakage and how `Pipeline` prevents it
> *Why must feature selection be performed inside the cross-validation loop, and how does a Pipeline prevent leakage?*

**Answer:**  
Feature selection utilizes statistical associations with target labels $y$. If selection is conducted on the whole training set prior to cross-validation, the validation fold labels directly influence feature selection. The validation folds are no longer independent holdouts, leading to overly optimistic performance estimates.  
A `Pipeline` strictly enforces modular boundaries: during each cross-validation fold, the selector and scaler execute `fit` and `transform` exclusively on the 4/5 training fold, applying only `transform` to the 1/5 validation fold, preventing any information flow from validation to training.

### Question 7: Decision guide across practical data science scenarios
> *In which situation would you choose each family of methods (thousands of features, small dataset, need for interpretability, limited computing time)?*

**Answer:**  
- **Thousands of features ($p \gg n$, e.g., genomics, transcriptomics, text):** Filter methods (ANOVA F, Mutual Information) or embedded $L_1$-regularization (Lasso). Wrappers are computationally intractable ($O(p^2)$) and overfit high-dimensional training splits.
- **Small datasets ($n$ small):** Filter methods or embedded methods; wrappers suffer high variance and severe overfitting on small validation splits.
- **Need for clinical/scientific interpretability:** Correlation filtering combined with RFE or Backward Elimination to produce a minimal, non-redundant set of distinct physical/biological measurements.
- **Limited computing time:** Filter methods (Variance Threshold + Correlation Filter + `SelectKBest`), which compute in milliseconds with linear $O(p)$ scaling.

---

## 8. Final Compliance Checklist

| Item | Requirement / Section | Status | Verification & Evidence |
|:----:|:----------------------|:------:|:------------------------|
| **1** | Repository structure & template cleanup | **PASS** | Root cleaned: `machine-learning-tp1-starter` and generic root `README.md` removed; TP5 in `TP5_Feature_Selection/`. |
| **2** | Part A: Exploration, target distribution, Dummy baseline | **PASS** | Shape (569, 30), 0 missing, Dummy accuracy 62.57%, `part_a_target_distribution.png` saved. |
| **3** | Part A: Stratified 70/30 split, Baseline Logistic Regression | **PASS** | Train acc 98.74%, Test acc 98.83%, Test $F_1$ 0.9907, train time 0.0086s. |
| **4** | Part A: Correlation matrix & Seaborn heatmap | **PASS** | `part_a_correlation_heatmap.png` generated; redundant groups analyzed. |
| **5** | Part B.1: Variance Threshold (0, 0.01, 0.1) on unscaled data | **PASS** | Removed feature counts reported; post-standardization invalidity explained. |
| **6** | Part B.2: Correlation removal (0.95, 0.90, 0.80) & contradiction documentation | **PASS** | `remove_correlated` implemented; CV validation and testing accuracies reported; methodology note documented. |
| **7** | Part B.3: ANOVA F, Mutual Information, Spearman rank correlation | **PASS** | Spearman $r = 0.9657$; sorted bar charts saved (`part_b_anova_f_scores.png`, `part_b_mutual_info_scores.png`). |
| **8** | Part B.3: `SelectKBest` across $k \in \{2..30\}$ with 1-SE rule | **PASS** | Accuracy vs. $k$ plotted (`part_b_selectkbest_accuracy_vs_k.png`); $k=20$ selected for both. |
| **9** | Part C.1: SFS Forward & Manual Forward Implementation | **PASS** | Both implemented; 10/10 overlap verified; trajectory plotted (`part_c_forward_selection.png`). |
| **10** | Part C.2: SFS Backward & Manual Backward Implementation | **PASS** | Both implemented; trajectory plotted (`part_c_backward_elimination.png`); divergence explained. |
| **11** | Part C.3: RFE, RFECV, and estimator comparison (SVC & RF) | **PASS** | RFECV optimal $k=8$; trajectory plotted (`part_c_rfecv_accuracy_vs_features.png`); estimator differences explained. |
| **12** | Part C.4: Wrapper cost formula & exhaustive search comparison | **PASS** | Formula $\sum (p-i) = 255$ subsets (1,275 models) verified against $2^{30}-1 \approx 1.07 \times 10^9$ subsets. |
| **13** | Part D.1: Noise injection (20 Gaussian features, $N=50$) | **PASS** | SelectKBest (0 noise), RFE (0 noise), Forward (2 noise) evaluated and analyzed. |
| **14** | Part D.2: Cross-validation leakage experiment | **PASS** | Leaky vs. Pipeline non-leaky CV compared; leakage mechanism and $p \gg n$ implications explained. |
| **15** | Part D.3: Final test evaluation across all 8 required pipelines | **PASS** | Exact requested columns reported; saved to `final_comparison_results.csv`. |
| **16** | Part D.3: Feature presence matrix & consensus features | **PASS** | Heatmap generated (`part_d_feature_presence_matrix.png`); consensus features listed (`worst texture`, `worst perimeter`). |
| **17** | Final Discussion: All 7 questions answered with results | **PASS** | Comprehensive answers addressing significance, computational cost, greedy nature, noise, leakage, and decision guide. |
| **18** | End-to-end execution & standalone deliverable | **PASS** | `tp5.py` executes completely without errors or deprecation warnings in 118s. |
