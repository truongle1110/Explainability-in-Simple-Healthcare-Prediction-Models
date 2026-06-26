from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def apply_shap(
    model: Any,
    x_train: Any,
    x_test: Any,
    feature_names: list[str] | pd.Index | None = None,
    class_index: int = 1,
    max_background_samples: int = 100,
    max_explain_samples: int | None = 200,
    output_dir: str | Path = "shap_outputs",
    plot_name: str = "model",
    max_display: int = 12,
    save_plots: bool = True,
    show_plots: bool = False,
    random_state: int = 42,
    dpi: int = 150,
) -> dict[str, Any]:
    """Apply SHAP explainability to a fitted sklearn-style model.

    Parameters
    ----------
    model:
        A fitted estimator or GridSearchCV object. If `best_estimator_` exists,
        that estimator is used for SHAP.
    x_train:
        Training features used as SHAP background data. Pass the scaled training
        data when the model was trained on scaled values.
    x_test:
        Test/validation features to explain. Pass the same feature representation
        used by the model.
    feature_names:
        Names for each feature. Required when `x_train`/`x_test` are numpy arrays
        and recommended for readable plots.
    class_index:
        Class to explain for classifiers. For this diabetes project, `1` means
        the positive diabetes outcome.

    Returns
    -------
    dict
        Includes the explainer, raw SHAP output, selected class SHAP values,
        feature-importance dataframe, explained rows, and saved output paths.
    """
    shap = _import_shap()
    import matplotlib.pyplot as plt

    estimator = _unwrap_estimator(model)
    x_train_df = _as_dataframe(x_train, feature_names)
    x_test_df = _as_dataframe(x_test, feature_names)

    background = _sample_rows(
        x_train_df,
        max_samples=max_background_samples,
        random_state=random_state,
    )
    x_explain = _sample_rows(
        x_test_df,
        max_samples=max_explain_samples,
        random_state=random_state,
    )

    explainer = _build_explainer(shap, estimator, background)
    shap_output = _calculate_shap_values(explainer, x_explain)
    shap_values = _select_class_values(shap_output, class_index=class_index)

    feature_importance = (
        pd.DataFrame(
            {
                "feature": x_explain.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    output_paths: dict[str, str] = {}
    if save_plots:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary_path = output_path / f"{plot_name}_shap_summary.png"
        shap.summary_plot(
            shap_values,
            x_explain,
            feature_names=list(x_explain.columns),
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(summary_path, dpi=dpi, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close()
        output_paths["summary_plot"] = str(summary_path)

        bar_path = output_path / f"{plot_name}_shap_bar.png"
        shap.summary_plot(
            shap_values,
            x_explain,
            feature_names=list(x_explain.columns),
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(bar_path, dpi=dpi, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close()
        output_paths["bar_plot"] = str(bar_path)

        importance_path = output_path / f"{plot_name}_shap_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        output_paths["feature_importance"] = str(importance_path)

    return {
        "estimator": estimator,
        "explainer": explainer,
        "raw_shap_output": shap_output,
        "shap_values": shap_values,
        "x_explain": x_explain,
        "feature_importance": feature_importance,
        "output_paths": output_paths,
    }


def _import_shap() -> Any:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install it with `pip install shap` or run "
            "`pip install -r dataset_processing/requirements.txt` after adding "
            "`shap>=0.44.0` to that file."
        ) from exc
    return shap


def _unwrap_estimator(model: Any) -> Any:
    return getattr(model, "best_estimator_", model)


def _as_dataframe(data: Any, feature_names: list[str] | pd.Index | None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("SHAP input data must be a 2D table of features.")

    if feature_names is None:
        columns = [f"feature_{idx}" for idx in range(values.shape[1])]
    else:
        columns = list(feature_names)
        if len(columns) != values.shape[1]:
            raise ValueError(
                f"Expected {values.shape[1]} feature names, got {len(columns)}."
            )

    return pd.DataFrame(values, columns=columns)


def _sample_rows(
    data: pd.DataFrame,
    max_samples: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_samples is None or max_samples >= len(data):
        return data.reset_index(drop=True)

    return data.sample(n=max_samples, random_state=random_state).reset_index(drop=True)


def _build_explainer(shap: Any, estimator: Any, background: pd.DataFrame) -> Any:
    if _looks_like_tree_model(estimator):
        try:
            return shap.TreeExplainer(estimator)
        except Exception:
            pass

    if hasattr(estimator, "coef_"):
        try:
            return shap.LinearExplainer(estimator, background)
        except Exception:
            pass

    try:
        return shap.Explainer(estimator, background)
    except Exception:
        if hasattr(estimator, "predict_proba"):
            return shap.Explainer(estimator.predict_proba, background)
        if hasattr(estimator, "predict"):
            return shap.Explainer(estimator.predict, background)
        raise


def _looks_like_tree_model(estimator: Any) -> bool:
    model_name = estimator.__class__.__name__.lower()
    return "forest" in model_name or "tree" in model_name or hasattr(
        estimator, "estimators_"
    )


def _calculate_shap_values(explainer: Any, x_explain: pd.DataFrame) -> Any:
    try:
        return explainer(x_explain)
    except Exception:
        return explainer.shap_values(x_explain)


def _select_class_values(shap_output: Any, class_index: int) -> np.ndarray:
    if isinstance(shap_output, list):
        return np.asarray(shap_output[class_index])

    values = np.asarray(getattr(shap_output, "values", shap_output))
    if values.ndim == 2:
        return values

    if values.ndim == 3:
        if values.shape[2] > class_index:
            return values[:, :, class_index]
        if values.shape[0] > class_index:
            return values[class_index]

    raise ValueError(
        "Could not select SHAP values for the requested class. "
        f"Received SHAP values with shape {values.shape}."
    )
