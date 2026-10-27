# Machine Learning Course Assignment

**Proposed topic:** Task 2, text classification with the IMDb Large Movie Review Dataset. Confirm the dataset with the instructor before treating this choice as final, as required by the assignment.

## Course information

- Course: Machine Learning
- Course code: confirm from the course LMS
- Semester: I, Academic Year 2026–2027
- Instructor: confirm from the course LMS
- Group members: see the report title page and contribution table; add each member's email before submission.

## Project structure

```text
notebooks/   Save/export the Google Colab front end here
modules/     Reusable Python code imported by the notebook
features/    Extracted text embeddings and metadata (.npy/.h5)
reports/     Final PDF report and exported figures
report/      LaTeX sources for the PDF report
doc/         Course assignment PDF
```

## Execution guide

1. Get instructor approval for the IMDb dataset (or update the project/report if another dataset is approved).
2. Create or upload the project notebook in Google Colab, then save/export its `.ipynb` file into `notebooks/`. Use a runtime with enough RAM; a GPU is recommended for DistilBERT feature extraction.
3. The notebook should install its documented dependencies, download the public dataset archive into the temporary Colab runtime, and extract it there. Do not mount Google Drive or another personal cloud drive.
4. Use **Runtime → Run all**. Confirm that EDA, preprocessing, TF-IDF creation, embedding extraction, classifier training, evaluation, and `.npy` export all complete without errors.
5. Preserve the official test split. Fit vectorizers and tune parameters using training/validation data only; use the test split once for final comparison.
6. Download generated embedding arrays, metadata, and final plots; place them under `features/` and `reports/`, and include them in the submission zip or provide clear download links.
7. Complete the report with measured results, actual contribution percentages, member emails, course code, instructor, and Colab link.

## Dataset and planned comparison

The proposed dataset has 50,000 labeled reviews with 25,000 examples each in the official train and test splits, balanced across positive and negative sentiment. Stanford's official dataset page and direct archive are:

- Dataset description: <https://ai.stanford.edu/~amaas/data/sentiment/>
- Archive: <https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz>

Compare traditional word/bigram TF-IDF features with frozen contextual DistilBERT embeddings. Save the DistilBERT representations and labels as `.npy` files. Compare Logistic Regression and linear SVM on both representations, with Multinomial Naive Bayes as an additional TF-IDF baseline. Report accuracy, macro precision, macro recall, macro F1, and confusion matrices. This is a plan; it does not claim any experiment has been completed.

## Submission checklist

- [ ] Colab notebook runs from a fresh runtime with **Run all** and no personal-drive mounts.
- [ ] Public data URL, installation, download, and extraction steps are automatic and documented.
- [ ] EDA covers class counts, word/character counts, text-length distributions, and frequent words.
- [ ] Both traditional and modern feature extraction are implemented and compared.
- [ ] Modern embeddings and labels are saved as `.npy` or `.h5`.
- [ ] PDF contains EDA, pipeline, experiments, analysis, figures/tables, and actual work distribution percentages.
- [ ] Submission zip contains `notebooks/`, `modules/`, `reports/`, and `features/`.
- [ ] Team evidence is included if collected; GitHub and README links are added if pursuing the bonus.
