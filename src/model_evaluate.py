import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, auc
import seaborn as sns
import numpy as np
import shap
from icecream import ic

class ModelEvaluator:
    def __init__(self,y_pred,y_pred_proba,y_test,X_test,random_search,categorical_cols,numerical_cols):
        self.y_test = y_test
        self.X_test = X_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.best_model = random_search.best_estimator_

    def evaluate(self):

        print("Classification Report:\n", classification_report(self.y_test, self.y_pred))
        print(self.__analyse_fail_rate())
        self.__plot_feature_importance()
        self.__plot_shap_summary()
        self.__plot_confusion_matrix()

    def __plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred, labels=[0,1])
        cm_percent = cm / cm.sum() * 100

        labels = np.array([f"{v}\n({p:.1f}%)" for v, p in zip(cm.flatten(), cm_percent.flatten())])
        labels = labels.reshape(cm.shape)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=labels, fmt='', cmap="magma", cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Real')
        plt.title('Confusion Matrix')
        plt.show()


    def __plot_feature_importance(self):
        encoder = self.best_model.named_steps['preprocess'].transformers_[0][1] 
        ohe_columns = encoder.get_feature_names_out(self.categorical_cols)
        all_columns = list(ohe_columns) + self.numerical_cols

        model = self.best_model.named_steps['model']

        importances = model.feature_importances_

        plt.figure(figsize=(10, 6))
        plt.barh(all_columns, importances, color='#f44f39')
        for index, value in enumerate(importances):
            plt.text(value, index, f'{value:.2f}', va='center', ha='left')

        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()

    # def __analyse_fail_rate(self):
    #     y_test_dist = pd.Series(self.y_test).value_counts(normalize=True)
    #     y_pred_dist = pd.Series(self.y_pred).value_counts(normalize=True)

    #     y_test_rate = y_test_dist[1]
    #     y_pred_rate = y_pred_dist[1]

    #     impact = (y_pred_rate - y_test_rate) * 100
    #     if impact >= 0:
    #         conclusion = (f"The model increased the fail rate by +{impact:.1f}% compared to the actual base.") #TODO
    #     else:
    #         conclusion = (f"The model reduced the fail rate by {abs(impact):.1f}% compared to the actual base.")   #TODO
    #     return conclusion
    
    def __analyse_fail_rate(self):
        """
        Analyze the difference in failure rate between the model's predictions and the actual test set.

        This function calculates the percentage of predicted failures and compares it to the true failure rate in the test set.
        It then returns a conclusion about whether the model is more or less sensitive to failures compared to the real data.

        Returns:
            str: A message describing the change in predicted failure rate relative to the actual rate.
        """
        y_test_dist = pd.Series(self.y_test).value_counts(normalize=True)
        y_pred_dist = pd.Series(self.y_pred).value_counts(normalize=True)

        y_test_rate = y_test_dist.get(1, 0)
        y_pred_rate = y_pred_dist.get(1, 0)

        impact = (y_pred_rate - y_test_rate) * 100
        if impact > 0:
            conclusion = (
                f"The model predicted {impact:.1f}% more failures than the actual base rate, "
                f"potentially increasing sensitivity to early signs of failure."
            )
        elif impact < 0:
            conclusion = (
                f"The model predicted {abs(impact):.1f}% fewer failures than the actual base rate, "
                f"which may indicate under-detection of failure conditions."
            )
        else:
            conclusion = (
                "The model predicted failures at the same rate as the actual base, indicating neutral alignment with reality."
            )
        return conclusion

    def __plot_shap_summary(self):
        preprocessor = self.best_model.named_steps['preprocess']
        X_test_transformed = preprocessor.transform(self.X_test)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(self.best_model.named_steps['model'])

        shap_values = explainer.shap_values(X_test_transformed)
        shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)
        shap.summary_plot(shap_values[1], X_test_transformed, feature_names=feature_names, plot_type='dot')