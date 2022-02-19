import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.metrics import classification_report, make_scorer

from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import fbeta_score, roc_auc_score

import pickle

import gc

import time
from contextlib import contextmanager

from tqdm import tqdm
tqdm.pandas()


# Timer
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print(f'{title} - done in {time.time() - t0:.0f}s')


# Define scorer from the fbeta score
fbeta_scorer = make_scorer(fbeta_score, beta=10)


# Cross-validated scores with up to four sampling methods and for two different metrics
def cv_scorer(
    kernel,
    X_train, y_train,
    X_test, y_test,
    methods=['none', 'undersampling', 'oversampling', 'balanced']
):
    # Outputs
    cv_results = pd.DataFrame(
        columns=[
            'train_ROC_AUC_mean_score',
            'test_ROC_AUC_score',
            'train_Fbeta_score',
            'test_Fbeta_score'
        ]
    )
    cv_classification_reports = {}
    fit_models = {}
    scoring = {
        'ROC_AUC_score': 'roc_auc',
        'Fbeta_score': fbeta_scorer
    }
    # Preprocessing pipeline
    preprocessor = make_column_transformer(
        (
            StandardScaler(),
            make_column_selector(dtype_include=np.number)
        ),
        (
            OneHotEncoder(handle_unknown='ignore'),
            make_column_selector(dtype_include='object')
        )
    )

    model = kernel['model']
    pipeline = make_pipeline(preprocessor, model)

    for method in tqdm(methods):
        if method == 'none':
            name = kernel['name']
            X_train_sampled, y_train_sampled = X_train, y_train
        elif method == 'undersampling':
            name = kernel['name_us']
            X_train_sampled, y_train_sampled = RandomUnderSampler(
                random_state=0).fit_resample(X_train, y_train)
        elif method == 'oversampling':
            name = kernel['name_os']
            X_train_sampled, y_train_sampled = SMOTENC(
                categorical_features=X_train.dtypes.eq('object'),
                random_state=0
            ).fit_resample(X_train, y_train)
        elif method == 'balanced':
            name = kernel['name_bal']
            model = kernel['model_bal']
            pipeline = make_pipeline(preprocessor, model)
            X_train_sampled, y_train_sampled = X_train, y_train

        print(name)
        with timer(name):
            # Calcute the cross-validation scores
            cross_val_scores = cross_validate(
                pipeline,
                X_train_sampled,
                y_train_sampled,
                cv=StratifiedKFold(n_splits=5),
                scoring=scoring,
                error_score='raise',
                verbose=4
            )
            # Fit the model and predict
            pipeline.fit(X_train_sampled, y_train_sampled)
            # Delete here the training set in order to optimize memory usage
            del X_train_sampled, y_train_sampled
            gc.collect()
            # Make the predictions
            y_pred = pipeline.predict(X_test)
            # Store the scores
            cv_results.loc[method] = [
                np.mean(cross_val_scores['test_ROC_AUC_score']),
                roc_auc_score(y_test, y_pred),
                np.mean(cross_val_scores['test_Fbeta_score']),
                fbeta_score(y_test, y_pred, beta=10)
            ]
            # Classification report
            cv_classification_reports[method] = classification_report(
                y_test, y_pred,
                target_names=['Non-defaulter', 'Defaulter'],
                zero_division=0
            )
            print(cv_classification_reports[method])

            ConfusionMatrixDisplay.from_estimator(
                pipeline,
                X_test, y_test,
                display_labels=['Non-defaulter', 'Defaulter']
            )
            plt.tight_layout()
            plt.show()
            PrecisionRecallDisplay.from_estimator(
                pipeline,
                X_test, y_test
            )
            plt.tight_layout()
            plt.show()
            RocCurveDisplay.from_estimator(
                pipeline,
                X_test, y_test
            )
            plt.plot(
                [0, 1], [0, 1],
                color='red',
                ls='--',
                lw=1.5
            )
            plt.tight_layout()
            plt.show()
            # Keep the fit model
            fit_models[method] = model

    # Save fit models, reports, and results
    pickle.dump(
        fit_models,
        open(kernel['fit_models_path'], 'wb')
    )
    pickle.dump(
        cv_classification_reports,
        open(kernel['cv_classification_reports_path'], 'wb')
    )
    cv_results.to_csv(kernel['cv_results_path'], index=False)

    return cv_results
