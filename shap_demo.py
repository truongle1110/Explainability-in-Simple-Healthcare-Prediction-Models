import matplotlib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shap_explainability import apply_shap


matplotlib.use("Agg")


def run_demo() -> None:
    data = pd.read_csv("dataset/diabetes.csv")

    target = "Outcome"
    x = data.drop(target, axis=1)
    y = data[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(random_state=100, max_iter=1000)
    model.fit(x_train_scaled, y_train)

    y_predict = model.predict(x_test_scaled)
    print(f"Demo accuracy: {accuracy_score(y_test, y_predict):.4f}")
    print(classification_report(y_test, y_predict))

    shap_result = apply_shap(
        model,
        x_train_scaled,
        x_test_scaled,
        feature_names=x.columns,
        output_dir="shap_outputs/logistic_regression_demo",
        plot_name="logistic_regression_demo",
        max_explain_samples=100,
        show_plots=False,
    )

    print("Top SHAP features:")
    print(shap_result["feature_importance"].head(8).to_string(index=False))
    print("Saved outputs:")
    for name, path in shap_result["output_paths"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    run_demo()
