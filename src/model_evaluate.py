import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import shap


class ModelEvaluator:
    def __init__(
        self,
        y_pred,
        y_pred_proba,
        y_test,
        X_test,
        random_search,
        categorical_cols,
        numerical_cols,
    ):
        self.y_test = y_test
        self.X_test = X_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.best_model = random_search.best_estimator_

    def evaluate(self):

        print(
            "Classification Report:\n", classification_report(self.y_test, self.y_pred)
        )
        self.__plot_feature_importance()
        self.__plot_shap_summaries()
        self.__plot_confusion_matrix()

    def __plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred, labels=[0, 1])
        cm_percent = cm / cm.sum() * 100

        labels = np.array(
            [f"{v}\n({p:.1f}%)" for v, p in zip(cm.flatten(), cm_percent.flatten())]
        )
        labels = labels.reshape(cm.shape)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=labels, fmt="", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Real")
        plt.title("Confusion Matrix")
        plt.show()

    def __plot_feature_importance(self):
        """Plots the top 10 most important features from the trained model."""
        encoder = self.best_model.named_steps["preprocess"].transformers_[0][1]
        ohe_columns = encoder.get_feature_names_out(self.categorical_cols)
        all_columns = list(ohe_columns) + self.numerical_cols

        model = self.best_model.named_steps["model"]
        importances = model.feature_importances_

        feature_importance = (
            pd.DataFrame({"feature": all_columns, "importance": importances})
            .sort_values(by="importance", ascending=False)
            .head(10)
        )

        plt.figure(figsize=(10, 6))
        plt.barh(
            feature_importance["feature"],
            feature_importance["importance"],
            color="#d3436e",
        )
        for index, value in enumerate(feature_importance["importance"]):
            plt.text(value, index, f"{value:.2f}", va="center", ha="left")

        plt.title("Top 10 Most Important Features")
        plt.xlabel("Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def __plot_shap_summaries(self):
        """Plots SHAP summary plots for the model's predictions."""
        preprocessor = self.best_model.named_steps["preprocess"]
        model = self.best_model.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()

        X_test_transformed = preprocessor.transform(self.X_test)
        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_transformed)

        shap.summary_plot(
            shap_values,
            X_test_transformed,
            feature_names=feature_names,
            plot_type="bar",
        )
        shap.summary_plot(
            shap_values[1],
            X_test_transformed,
            feature_names=feature_names,
            plot_type="dot",
        )
