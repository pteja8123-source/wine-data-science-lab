# Wine data science lab

This AI-assisted learning project uses the UCI Wine Recognition dataset to practice Python statistics, data cleaning, plotting, principal component analysis and multiclass classification.

## Run

```sh
python -m venv .venv
python -m pip install -r requirements.txt
python analyze.py
```

Activate the virtual environment before installing. On Windows Command Prompt, run `.venv\Scripts\activate.bat`. In Jupyter, use `%run analyze.py` from the repository folder in a kernel with the listed packages.

The script prints the data checks and model metrics and creates three plots and a group summary in `results/`. On the fixed stratified split, logistic regression classified 45 of 45 held-out samples correctly. The majority baseline got 18 of 45. A single split on a small benchmark gives limited evidence of performance on new samples.

Data source: [UCI Wine Recognition](https://doi.org/10.24432/C5PC7J), Aeberhard and Forina, 1992, CC BY 4.0. The script loads the dataset through scikit-learn.
