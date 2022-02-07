import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

from sklearn.metrics import classification_report, make_scorer

from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay,\
    PrecisionRecallDisplay,\
    RocCurveDisplay
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


# # Loan repayment metric
# def loan_repayment_score(y_true, y_pred, a=1, b=-1, c=-5, d=0):
#     (tn, fp, fn, tp) = confusion_matrix(y_true, y_pred).ravel()

#     assert (abs(c) > abs(b) >= abs(a) > abs(d)),\
#         'The four coefficients of the functional must satisfy |c| > |b| >= |a| > |d|.'

#     gain_tot = abs(a)*tn - abs(b)*fp - abs(c)*fn + abs(d)*tp
#     gain_max = (tn + fp)*abs(a) + (fn + tp)*abs(d)
#     gain_min = -(tn + fp)*abs(b) - (fn + tp)*abs(c)

#     return (gain_tot - gain_min) / (gain_max - gain_min)


# # Define scorer from the loan repayment score
# loan_repayment_scorer = make_scorer(loan_repayment_score)


# Define scorer from the fbeta score
fbeta_scorer = make_scorer(fbeta_score, beta=10)


# Cross-validated scores with up to four sampling methods and for two different metrics
def cv_scorer(
    kernel,
    X_train_prep, y_train_prep,
    X_test_prep, y_test,
    methods=['none', 'undersampling', 'oversampling', 'balanced']
):
    cv_results = pd.DataFrame(
        columns=[
            'balancing',
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

    for i, method in enumerate(tqdm(methods)):
        model = kernel['model']

        if method == 'none':
            name = kernel['name']
            X_train, y_train = X_train_prep, y_train_prep
        elif method == 'undersampling':
            name = kernel['name_us']
            X_train, y_train = RandomUnderSampler(
                random_state=0).fit_resample(X_train_prep, y_train_prep)
        elif method == 'oversampling':
            name = kernel['name_os']
            X_train, y_train = SMOTE(random_state=0).fit_resample(
                X_train_prep, y_train_prep)
        elif method == 'balanced':
            model = kernel['model_bal']
            name = kernel['name_bal']
            X_train, y_train = X_train_prep, y_train_prep

        print(name)
        with timer(name):
            # Calcute the cross-validation scores
            cross_val_scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=StratifiedKFold(n_splits=5),
                scoring=scoring,
                error_score='raise'
            )
            # Fit the model and predict
            model.fit(X_train, y_train)
            # Delete here the training set in order to optimize memory usage
            del X_train, y_train
            gc.collect()
            # Make the predictions
            y_pred = model.predict(X_test_prep)
            # Store the scores
            cv_results.loc[i] = [
                method,
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
                model,
                X_test_prep, y_test,
                display_labels=['Non-defaulter', 'Defaulter']
            )
            plt.tight_layout()
            plt.show()
            PrecisionRecallDisplay.from_estimator(
                model,
                X_test_prep, y_test
            )
            plt.tight_layout()
            plt.show()
            RocCurveDisplay.from_estimator(
                model,
                X_test_prep, y_test
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
