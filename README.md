# Wine data science lab

This AI-assisted learning project uses the UCI Wine Recognition dataset to practice Python statistics, data cleaning, plotting, principal component analysis and multiclass classification.

## Run

```sh
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python analyze.py
```

The commands above are for Windows Command Prompt. To run the notebook in the same environment, install and start Jupyter from the repository root:

```cmd
python -m pip install notebook
python -m notebook wine_analysis.ipynb
```

Choose **Run All**. The notebook includes saved outputs and charts for readers who open it on GitHub.

## Where to find each task

| Task | Evidence in this repository |
| --- | --- |
| 1. Fundamentals | The data checks and statistics cell illustrates numeric measurements, categorical labels, lists, dictionaries, the mean, median, and sample standard deviation. |
| 2. Hands-On Data Lab | The setup commands, notebook, grouped CSV export, and class counts chart show the lab workflow. |
| 3. Dataset Analysis | The notebook checks for missing values and duplicates and compares flavanoids and color intensity by class. |
| 4. Tool Mastery | The notebook uses pandas for grouping, Matplotlib for plotting, and scikit-learn for scaling and PCA. |
| 5. Analytics Report | The class means and measurement scatter plot support the findings and recommendations in the separately submitted report. |
| 6. Capstone | The final notebook cell compares a majority-class baseline with logistic regression on held-out data and prints both confusion matrices. |

[Open the notebook](wine_analysis.ipynb). The individual Word reports were submitted separately through YuvaIntern.

The script prints the data checks and model metrics and creates three plots and a group summary in `results/`. On the fixed stratified split, logistic regression classified 45 of 45 held-out samples correctly. The majority baseline got 18 of 45. A single split on a small benchmark gives limited evidence of performance on new samples.

Data source: [UCI Wine Recognition](https://doi.org/10.24432/C5PC7J), Aeberhard and Forina, 1992, CC BY 4.0. The script loads the dataset through scikit-learn.
