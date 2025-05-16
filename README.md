# Dataset Integration Guide

This guide outlines the requirements for providing datasets to ensure seamless integration with our machine learning pipeline. The minimal working example notebook is `main.ipynb`, and you should implement your datasets in the `custom_datamodule`.

## Dataset Requirements for Pipeline Integration

To ensure seamless integration with our machine learning pipeline, which primarily utilizes XGBoost for modeling, please adhere to the following requirements when providing your datasets:

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

* To ensure data compatibility, we recommend running a baseline XGBoost model on your end using the provided skeleton pipeline code and comparing the results (e.g., average AUROC) with a run performed on our end. This helps quickly identify any potential discrepancies or issues with the data format or content.