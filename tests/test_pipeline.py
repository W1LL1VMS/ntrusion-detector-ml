import matplotlib

matplotlib.use("Agg")

import pandas as pd

from intrusion_detector.data import CATEGORIES, FEATURE_NAMES, split_features_target
from intrusion_detector.evaluate import (
    BINARY_SCORING,
    attack_score,
    binary_metrics_from_families,
    cross_val_predictions,
    cross_validate_binary,
    family_summary,
    per_class_report,
    plot_confusion_matrix,
)
from intrusion_detector.models import (
    BALANCED_SUFFIX,
    baseline_models,
    candidate_models,
    make_pipeline,
)
from intrusion_detector.preprocessing import (
    DROPPED_FEATURES,
    USED_NUMERIC_FEATURES,
    build_preprocessor,
)
from sklearn.linear_model import LogisticRegression


def test_preprocessor_output_width(sample_df):
    X, _ = split_features_target(sample_df)
    preprocessor = build_preprocessor().fit(X)

    transformed = preprocessor.transform(X)

    # variables numériques + 3 protocoles + 4 services + 3 drapeaux
    assert transformed.shape == (len(X), len(USED_NUMERIC_FEATURES) + 3 + 4 + 3)
    assert not pd.DataFrame(transformed).isna().any().any()


def test_constant_feature_is_dropped(sample_df):
    X, _ = split_features_target(sample_df)
    names = build_preprocessor().fit(X).get_feature_names_out()

    for dropped in DROPPED_FEATURES:
        assert dropped not in names


def test_unknown_category_is_ignored_instead_of_failing(sample_df):
    X, _ = split_features_target(sample_df)
    preprocessor = build_preprocessor().fit(X)

    new_row = X.iloc[[0]].copy()
    new_row["service"] = "service_jamais_vu"
    transformed = pd.DataFrame(preprocessor.transform(new_row), columns=preprocessor.get_feature_names_out())

    service_columns = [name for name in transformed.columns if name.startswith("service_")]
    assert len(service_columns) == 4
    assert (transformed[service_columns] == 0).all().all()


def test_pipeline_fits_and_predicts(sample_df):
    X, y = split_features_target(sample_df)
    pipeline = make_pipeline(LogisticRegression(max_iter=500)).fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions) <= {0, 1}


def test_baseline_always_predicts_the_majority_class(sample_df):
    X, y = split_features_target(sample_df)
    dummy = baseline_models()["Référence (classe majoritaire)"].fit(X, y)

    assert set(dummy.predict(X)) == {1}  # 40 attaques contre 10 connexions normales


def test_cross_validation_reports_every_metric(sample_df):
    X, y = split_features_target(sample_df)
    model = make_pipeline(LogisticRegression(max_iter=500))

    table = cross_validate_binary(model, X, y, n_splits=5, n_jobs=1)

    assert list(table.index) == list(BINARY_SCORING)
    assert list(table.columns) == ["moyenne", "écart-type"]
    assert table["moyenne"].between(0, 1).all()


def test_per_class_report_covers_every_family(sample_df):
    X, y = split_features_target(sample_df, target="category")
    model = make_pipeline(LogisticRegression(max_iter=500))

    y_true, y_pred = cross_val_predictions(model, X, y, n_splits=5, n_jobs=1)
    report = per_class_report(y_true, y_pred)

    assert list(report.index) == CATEGORIES + ["macro avg"]
    assert report.loc[CATEGORIES, "support"].sum() == len(sample_df)


def test_confusion_matrix_plot_runs(sample_df):
    y_true = sample_df["category"].astype(str)
    ax = plot_confusion_matrix(y_true, y_true, title="test")

    assert ax.get_title() == "test"


def test_candidate_models_come_with_and_without_class_weights():
    plain, weighted = candidate_models(), candidate_models(balanced=True)

    assert len(plain) == len(weighted) == 3
    assert not any(name.endswith(BALANCED_SUFFIX) for name in plain)
    assert all(name.endswith(BALANCED_SUFFIX) for name in weighted)


def test_every_candidate_model_fits_and_predicts(sample_df):
    X, y = split_features_target(sample_df, target="category")
    y = y.astype(str)

    for name, model in {**candidate_models(), **candidate_models(balanced=True)}.items():
        predictions = model.fit(X, y).predict(X)
        assert len(predictions) == len(X), name


def test_family_summary_has_one_row_per_model(sample_df):
    y = sample_df["category"].astype(str)
    naive = pd.Series(["normal"] * len(y), index=y.index)

    summary = family_summary({"parfait": (y, y), "naïf": (y, naive)})

    assert list(summary.index) == ["parfait", "naïf"]
    assert list(summary.columns) == CATEGORIES + ["F1 macro"]
    assert summary.loc["parfait", "F1 macro"] == 1.0
    assert summary.loc["naïf", "normal"] == 1.0
    assert summary.loc["naïf", "u2r"] == 0.0


def test_final_model_can_be_saved_and_reloaded(sample_df, tmp_path):
    from intrusion_detector.train import FINAL_MODEL_NAME, load_model, save_model, train_final_model

    X, _ = split_features_target(sample_df)
    model = train_final_model(sample_df)
    path = tmp_path / "models" / "final_model.joblib"

    save_model(model, path)
    reloaded = load_model(path)

    assert FINAL_MODEL_NAME.startswith("Gradient boosting")
    assert list(reloaded.predict(X)) == list(model.predict(X))
    assert set(reloaded.classes_) == set(CATEGORIES)


def test_loading_a_missing_model_explains_how_to_train_it(tmp_path):
    import pytest

    from intrusion_detector.train import load_model

    with pytest.raises(FileNotFoundError, match="intrusion_detector.train"):
        load_model(tmp_path / "absent.joblib")


def test_loading_a_model_from_another_sklearn_version_warns(sample_df, tmp_path):
    import joblib
    import pytest

    from intrusion_detector.train import load_model, train_final_model

    path = tmp_path / "old_model.joblib"
    joblib.dump({"model": train_final_model(sample_df), "name": "x", "sklearn_version": "0.0.1"}, path)

    with pytest.warns(UserWarning, match="scikit-learn"):
        load_model(path)


def test_binary_metrics_collapse_families_into_normal_or_attack():
    y_true = ["normal", "dos", "probe", "r2l"]
    y_pred = ["normal", "probe", "normal", "r2l"]  # une attaque est ratée, une autre change de famille

    metrics = binary_metrics_from_families(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0  # tout ce qui est signalé est bien une attaque
    assert abs(metrics["recall"] - 2 / 3) < 1e-9
    assert "roc_auc" not in metrics


def test_attack_score_is_one_minus_the_normal_probability(sample_df):
    from intrusion_detector.train import train_final_model

    X, _ = split_features_target(sample_df)
    model = train_final_model(sample_df)

    score = attack_score(model, X)
    normal_probability = model.predict_proba(X)[:, list(model.classes_).index("normal")]

    assert score.shape == (len(X),)
    assert ((score >= 0) & (score <= 1)).all()
    assert abs((score + normal_probability) - 1).max() < 1e-9
