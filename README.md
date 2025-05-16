# Dataset Integration Guide

Welcome! This guide explains how to prepare and provide your datasets for seamless integration with our machine learning pipeline, which ultimately uses our TabPFN model. To ensure compatibility, we'll use XGBoost for initial testing and benchmarking.

**Key Files & Workflow:**

1.  **Implement Your Dataset Loader:**
    * You'll primarily work in `src/data_processing/custom_datamodule.py`. This is where you'll implement the logic to load and structure your datasets according to the `BaseDataModule` concept (see the provided skeleton code).
2.  **Test Your Implementation:**
    * Use the `minimal_example.ipynb` notebook to test your dataset integration. You can modify this notebook or create a new one.
    * This notebook runs XGBoost. Since XGBoost shares a similar scikit-learn interface with our TabPFN model, this helps ensure smooth integration with TabPFN. It also allows you to compare results with your existing codebase if you also use XGBoost.
3.  **Run Automated Checks:**
    * Execute `pytest` to run automated tests. These tests quickly verify if your data is in the correct format.
4.  **Storing Data:**
    * A `data/` folder is included in the repository for your datasets.
    * If your datasets are too large for the repository, please provide a download link or clear instructions on how to access them.

**Customization (If Needed):**

* **Result Aggregation:** If you require specific ways to aggregate results across datasets (beyond the default average), implement your logic in `src/utils.py`. The `benchmark_datasets()` function currently averages metrics.
* **Additional Metrics:** If you need metrics not already included, you can add them by editing `MetricType` in `src/metrics.py`.

## Dataset Requirements for Pipeline Integration

To ensure your datasets work flawlessly with our pipeline, please follow these requirements. We've marked requirements as **(Essential)** or **(Nice-to-have)**.


### 1. Data Structure & Format: (Essential)

* **Data Provision:** Please provide data structured similarly to the `BaseDataModule` concept (see provided code skeleton). This means supplying separate lists for training, testing, and optionally, validation datasets.
* **Dataset Lists:**
    * Each list (train, test, validation) should contain individual datasets.
    * Each dataset within these lists must be provided as a pair of NumPy arrays:
        * `X`: Feature matrix (`np.ndarray`).
        * `y`: Target vector (`np.ndarray`).
* **Array Format:**
    * `X` arrays should have a shape of `(n_samples, n_features)`.
    * `y` arrays should have a shape of `(n_samples,)`.
    * Data types should be suitable for direct use in XGBoost (typically numerical types like `float` or `int`).

### 2. Data Content & Quality:

* **Pre-processing (Essential):** Datasets should be pre-processed and ready for direct model input. No additional feature selection should be required on our end.
* **No Leakage (Essential):** Ensure that test sets (and validation sets, if provided) contain no information leaked from the training set (e.g., target information encoded in features, samples present in both train and test).
* **Shape Constraints (Essential):** Individual datasets (`X` arrays) should ideally not exceed **10,000 samples** & **500 features** for validation and test, and **5,000 samples** & **500 features** for training. If larger datasets are necessary, please discuss this with us beforehand.
* **Feature Consistency (Nice to have):**
    * **Consistent Pre-processing:** Applying the same pre-processing steps across all datasets would be beneficial.
    * **Related Tables:** Datasets should ideally be related, for example, sharing a similar domain or feature set.

### 3. Metadata (Nice to have):

* Please provide the following metadata for **each** set of datasets (train/test/validation lists):
    * `feature_names`: A list of strings corresponding to the columns in the `X` arrays (required if features are consistent).

### 4. Splitting & Evaluation (Essential):

* **Train/Test Sets:** Providing distinct training and testing lists is mandatory.
* **Validation Sets (Optional):** You may provide a separate list of validation datasets. If not provided, we will create validation splits by randomly sampling from the training data.
* **Time Series Data:** If dealing with time series, please provide a custom splitting function (`splitfn`) that respects the temporal order of the data to avoid leakage during cross-validation or train/test splits. Discuss this requirement with us.
* **Evaluation Metrics:**
    * Specify the primary metric(s) you care about for evaluating model performance (e.g., Accuracy, AUROC, LogLoss, F1-score).
    * Define the **final success metric**: How should performance across multiple test datasets be aggregated? (e.g., simple average AUROC, weighted average based on dataset size, etc.). If you have a specific function for this calculation, please provide it. Ideally, these can be reduced to a single number to be optimized.

### 5. Verification (Recommended):

* To ensure data compatibility, we recommend running a baseline XGBoost model on your end using the provided skeleton pipeline code and comparing the results (e.g., average AUROC) with a run performed on your existing codebase. This helps quickly identify any potential discrepancies or issues with the data format or content.