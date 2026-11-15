# Machine Learning Course Assignment

**Selected topic:** Task 2, five-class sentiment classification with the Coronavirus Tweets NLP dataset using TF-IDF and pretrained GloVe embeddings.

## Course information

- Course: Machine Learning
- Course code: CO3117
- Semester: I - Academic Year 2026–2027
- Instructor: Trương Vĩnh Lân

## Project structure

```text
notebooks/   Single Google Colab front end
modules/     Reusable Python code imported by the notebook
features/    Generated feature matrices, aligned labels, and metadata
outputs/     Generated figures, metrics, audits, and fitted models
reports/     Final compiled PDF report
report/      LaTeX sources for the PDF report
doc/         Course assignment PDF
```

Downloaded datasets, serialized models, bulky feature arrays, LaTeX intermediates, and local virtual environments are intentionally excluded from Git. Compact metadata plus report tables and figures remain versionable for reproducibility. The canonical compiled report is `reports/Final_Report.pdf`; `report/report.pdf` is only a local LaTeX build artifact.

## Execution guide

1. Open `notebooks/task2_pipeline.ipynb` from the extracted project folder in Google Colab.
2. Use **Runtime → Run all**. The notebook installs dependencies, downloads and verifies both CSV files, runs EDA, extracts TF-IDF and GloVe-Twitter-100 features, trains Logistic Regression and linear SVM on both representations, and exports the measured results automatically. The first run also downloads the pretrained GloVe vectors. Do not mount Google Drive or another personal cloud drive.
3. Detailed EDA must use the official training split only. Preserve the official test split for final evaluation; fit vectorizers and tune parameters using training/validation data only.
4. Build the LaTeX report under `report/`, then copy only the final submission PDF into `reports/`.
5. Before submission, confirm actual contribution percentages, member emails, course code, instructor, and the Colab link.

For local execution, use Python 3.10--3.13, create a virtual environment, and install `requirements.txt` (Gensim does not currently build on Python 3.14). The notebook is the canonical entry point; its EDA section replaces the former standalone EDA notebook.

## Dataset and planned comparison

The selected dataset contains 41,157 training tweets and 3,798 test tweets labelled as Extremely Negative, Negative, Neutral, Positive, or Extremely Positive.

- Original dataset page: <https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification>
- Reproducible notebook download: pinned public copies of the two original CSV files, with SHA-256 verification, are configured in `modules/data_downloader.py`.

The implemented EDA audits missing values, duplicate and conflicting texts, train/test overlap, class distribution, tweet length, URLs, mentions, hashtags, and frequent unigrams/bigrams. It saves report-ready figures under `outputs/graphics/eda/` and tables under `outputs/generated/eda/`. After feature extraction, a shared stratified training sample is also projected with PCA/SVD, t-SNE, and UMAP to create six feature-space figures under `outputs/graphics/feature_space/`.

The controlled experiment uses Logistic Regression and linear SVM with both representations. TF-IDF linear SVM achieved 56.69% accuracy and 0.5807 macro F1; TF-IDF Logistic Regression achieved 56.90% accuracy and 0.5728 macro F1. Mean-pooled GloVe-Twitter-100 was weaker: Logistic Regression achieved 42.36% accuracy and 0.4294 macro F1, while linear SVM achieved 41.92% accuracy and 0.4272 macro F1. All values come from the untouched official test set.

## Implementation map

- `modules/data_downloader.py`: pinned dataset downloads, checksum verification, and schema validation.
- `modules/text_eda.py`: quality auditing, descriptive statistics, and EDA exports.
- `modules/data_preprocessor.py`: shared leakage-safe splitting and text normalization.
- `modules/feature_extractor.py`: interchangeable TF-IDF and GloVe extractors with the same `fit`/`transform` interface.
- `modules/models.py`: interchangeable Logistic Regression and linear-SVM classes with the same `fit`/`predict` interface.
- `modules/trainer.py`: model-agnostic hyperparameter tuning and final model fitting.
- `modules/evaluator.py`: validation/test metrics, model selection, and evaluation-artifact export.
- `modules/plotter.py`: every EDA, feature-space, confusion-matrix, and comparison plot.
- `features/`: sparse TF-IDF matrices, dense GloVe arrays, aligned labels, and metadata.
- `outputs/generated/test_metrics.csv`: the single consolidated model-results table.
- `outputs/generated/eda/`: two compact EDA summary tables and metadata.
- `outputs/graphics/`: report-ready EDA figures, six feature-space projections, selected confusion matrices, and the final comparison chart.
- `outputs/models/`: serialized fitted pipelines grouped by feature extractor.
- `reports/Final_Report.pdf`: the only versioned compiled report deliverable.