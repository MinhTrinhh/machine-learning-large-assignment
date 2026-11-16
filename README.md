# COVID-19 Tweet Sentiment Classification

Machine Learning course assignment comparing TF-IDF and pretrained GloVe-Twitter features for five-class sentiment classification with Logistic Regression and linear SVM.

## Course information

| Item | Details |
|---|---|
| University | VNU-HCM University of Technology |
| Faculty | Computer Science and Engineering |
| Course | Machine Learning (CO3117) |
| Semester | I — Academic Year 2026–2027 |
| Instructor | Trương Vĩnh Lân |
| Group | 15 |

## Group members

| Student | Student ID |
|---|---:|
| Trịnh Lê Minh | 2352765 |
| Trần Phước Sang | 2353044 |

## Project links

- [Open the executable notebook in Google Colab](https://colab.research.google.com/github/MinhTrinhh/machine-learning-large-assignment/blob/main/notebooks/colab_pipeline.ipynb)
- [Read the final report](report/report.pdf)
- [View the GitHub repository](https://github.com/MinhTrinhh/machine-learning-large-assignment)

## Run the project

1. Open the Google Colab link above.
2. Select **Runtime → Run all**.
3. Wait for every cell to complete.

The notebook automatically:

- clones the supporting source code;
- installs the required Python libraries;
- downloads and verifies the public dataset;
- prepares the data and performs EDA;
- extracts TF-IDF and GloVe feature representations;
- trains and evaluates the configured models; and
- generates metrics, figures, feature files, and fitted models.

The first complete run also downloads the pretrained GloVe-Twitter-100 embeddings and may take longer than later runs. Colab storage is temporary, so download any artifacts you want to retain before closing the runtime.

## Repository structure

```text
notebooks/
  colab_pipeline.ipynb    Google Colab front end and complete execution workflow
modules/
  data_downloader.py      Public dataset download and integrity verification
  data_preprocessor.py    Text cleaning and leakage-safe data preparation
  text_eda.py             Exploratory data analysis
  feature_extractor.py    TF-IDF and GloVe feature extraction
  models.py               Logistic Regression and linear-SVM definitions
  trainer.py              Hyperparameter tuning and final training
  evaluator.py            Validation and test evaluation
  plotter.py              EDA, feature-space, and result visualizations
requirements.txt          Python dependencies installed by the notebook
features/                 Generated feature representations and metadata
outputs/                  Generated metrics, figures, audits, and fitted models
report/                   LaTeX sources and compiled assignment report
```

## Generated outputs

After execution, the main runtime artifacts are available under:

- `features/` — extracted feature matrices and aligned labels;
- `outputs/generated/` — metrics, cross-validation results, and audit files;
- `outputs/graphics/` — EDA figures, feature projections, comparisons, and confusion matrices; and
- `outputs/models/` — fitted model pipelines.
