# -Feature-Engineering-

# ML Feature Engineering Pipeline

This repository contains the Python code for a university assignment demonstrating a complete feature engineering pipeline. The project applies various preprocessing, feature extraction, and selection techniques to two distinct datasets:

1.  **Image Data (Computer Vision):** [Dog vs Cat Dataset](https://www.kaggle.com/datasets/anthonytherrien/dog-vs-cat)
2.  **Text Data (NLP):** [IMDB 50K Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

The project is implemented in a Jupyter Notebook / Google Colab environment using `tensorflow`, `pandas`, and `scikit-learn`.

---

## ⚙️ Feature Engineering Steps

### 1. Image Dataset (Dog vs. Cat)

For the image data, standard preprocessing steps were adapted for a deep learning context.

* **Load & Encode:** Data was loaded using `tf.keras.utils.image_dataset_from_directory`. This function efficiently loads images and automatically performs **Label Encoding** on the target variable by converting the folder names (`cat`, `dog`) into binary labels (`0`, `1`).
* **Handle Missing Data:** The loader **automatically skips** any corrupted or unreadable image files.
* **Feature Scaling:** We used `tf.keras.applications.mobilenet_v2.preprocess_input`. This is a specialized **Normalization** function that scales pixel values from the `[0, 255]` range to the `[-1, 1]` range required by the pre-trained model.
* **Feature Extraction (Transfer Learning):** Instead of PCA, we used a pre-trained **MobileNetV2** model as a feature extractor. By setting `include_top=False` and `trainable=False`, we use the model's powerful, pre-learned filters to convert raw images into high-level feature maps.
* **Feature Selection / Reduction:** A `GlobalAveragePooling2D` layer was added after the feature extractor. This layer "selects" the most dominant features from the multi-dimensional feature maps and reduces them to a single vector, making classification efficient.
* **Impact:** This pipeline transformed raw image files into a format suitable for a neural network. Using pre-trained features allowed our simple model to achieve **100% validation accuracy**, demonstrating the power of transfer learning as a feature engineering technique.

### 2. Text Dataset (IMDB Movie Reviews)

For the text data, a classic Natural Language Processing (NLP) pipeline was built.

* **Load & Explore:** Data was loaded into a `pandas` DataFrame. Exploration with `.info()` and `.isnull().sum()` confirmed there was **no missing data** to impute.
* **Data Cleaning:** A custom function was applied to all reviews to:
    1.  Remove HTML tags (`<br />`).
    2.  Remove punctuation and numbers.
    3.  Convert all text to lowercase.
    4.  Remove common English "stop words" (e.g., 'the', 'is', 'a').
* **Encode Categorical Variables:** The `sentiment` column ('positive'/'negative') was **Label Encoded** to `1` and `0` using `sklearn.preprocessing.LabelEncoder`.
* **Feature Extraction (TF-IDF):** The raw text was converted into a numerical matrix using `TfidfVectorizer`. This extracted **5,000 numerical features** based on word importance (Term Frequency-Inverse Document Frequency).
* **Feature Scaling:** The `TfidfVectorizer` **automatically performs L2 Normalization** on its output by default, which fulfills the scaling requirement.
* **Feature Selection:** `SelectKBest` with the `chi2` statistical test was applied to the 5,000 TF-IDF features to select the **top 2,000 most statistically relevant features**.
* **Impact:** The pipeline successfully transformed unstructured, "dirty" text into a clean, normalized, and feature-selected matrix. A Naive Bayes model trained on this final matrix achieved **~88% accuracy**, proving the transformations were highly effective.

---

## ⚖️ Ethical Discussion

The final part of the assignment discusses the ethical concerns of using protected-class features like 'Gender' or 'Marital Status' in a predictive model (e.g., for employee attrition). The primary mitigation strategy identified is the **exclusion of these features** from the model and the subsequent **auditing for proxy-variable bias** to ensure fairness.
