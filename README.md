# Iris Classification

Task 1 for InternSpark — building a model that predicts iris species from sepal/petal measurements.

## Files

- `Iris_Classification.ipynb` — main notebook, run top to bottom in Colab
- `iris.csv` — dataset (150 rows, same format as the Kaggle version)
- `iris_model.joblib` — saved model (generated after running the notebook)
- `scaler.joblib`, `label_encoder.joblib` — needed for inference

## How to run

1. Open `Iris_Classification.ipynb` in Google Colab
2. Upload `iris.csv` to the Colab session (Files panel on the left)
3. Run all cells

That's it, no API keys or downloads needed, the CSV is loaded locally.

## What's in the notebook

- Quick EDA — class counts, pairplot, correlation heatmap, boxplots per feature
- Trained and compared three models: k-NN, Logistic Regression, Decision Tree
- Evaluated with accuracy, precision, recall, and confusion matrices
- Saved the best-performing model with joblib

Petal length/width turned out to be the most useful features for telling the classes apart, sepal measurements overlap more between versicolor and virginica. All three models perform well on this dataset since the classes are fairly separable to begin with, but Logistic Regression and k-NN edged out the Decision Tree slightly on the test split.

## Running inference

```python
predict_species(5.1, 3.5, 1.4, 0.2)
```

This loads the saved model, scaler, and label encoder and returns the predicted species name. Note: if the Decision Tree ends up being the best model, skip the scaling step since it was trained on unscaled data.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
```

Packages come pre-installed in Colab, so no extra setup needed there.
