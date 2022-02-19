from dash import html, Input, Output
import dash_bootstrap_components as dbc

from app import app

import numpy as np
import pandas as pd

import pickle

from utils.functions import *


# Import modelling data
df = reduce_memory_usage(
    pd.read_csv(
        'data/fulfilled_data.csv',
        index_col='SK_ID_CURR'
    )
)
# Get a sample of the modelling testing data
X_test = df[df['TARGET'].isna()].drop(columns='TARGET').tail(100)
# Convert days to years
# NB These two new features will not affect the performance of the fit model
# since they will automatically not be taken into account
X_test['AGE_YEARS'] = X_test['DAYS_BIRTH'].div(-365).apply(np.round, decimals=2)
X_test['YEARS_EMPLOYED'] = X_test['DAYS_EMPLOYED'].div(-365).apply(np.round, decimals=2)
# From the full preprocessed testing data just the same observations are needed
# in order to run the model
X_test_prep = reduce_memory_usage(
    pd.read_csv(
        'data/fulfilled_data_prep.csv',
        index_col='SK_ID_CURR'
    ).loc[X_test.index]
)
X_test_prep.drop(columns='TARGET', inplace=True)

# Import the model
clf = pickle.load(open('model/logistic_cv_model.pkl', 'rb'))
log_reg = clf.best_estimator_
# Retrieve the model coefficients
feature_coef = pd.Series(
    log_reg.named_steps.logisticregression.coef_[0],
    index=log_reg.named_steps.columntransformer.get_feature_names_out()
).rename('coefficient')


# # Callbacks
# Loan score callback
@app.callback(
    [
        Output('gauge-score', 'figure'),
        Output('impact_pos', 'figure'),
        Output('impact_neg', 'figure'),
        Output('title', 'children'),
        Output('decision', 'children')
    ],
    [Input('sk-id-curr', 'value')]
)
def update_loan_score(ID):
    fig_pos, fig_neg = feature_impact(ID, X_test_prep, feature_coef)

    if ID:
        defaulter_class, prob = binary_prediction(ID, log_reg, X_test)
        fig = gauge_chart(prob)
        if defaulter_class == 0:
            msg = html.H3(
                '... could be granted',
                style={'color': 'green', 'textAlign': 'right'}
            )
        else:
            msg = html.H3(
                '... should be rejected!',
                style={'color': 'red', 'textAlign': 'right'}
            )
        title = html.H3(f'The loan {ID}...')
    else:
        fig = gauge_chart(.5)
        title = html.H3('Chances of the loan...')
        msg = html.H3(style={'textAlign': 'right'})

    return fig, fig_pos, fig_neg, title, msg


# Client details callback
@app.callback(
    Output('table', 'data'),
    [
        Input('sk-id-curr', 'value'),
        Input('buttons', 'value')
    ]
)
def update_loan_details(ID, val_button):
    client_info_cols = [
        'CODE_GENDER',
        'FLAG_OWN_CAR',
        'AMT_INCOME_TOTAL',
        'NAME_TYPE_SUITE',
        'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE',
        'AGE_YEARS',
        'YEARS_EMPLOYED',
        'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE'
    ]
    loan_info_cols = [
        'NAME_CONTRACT_TYPE',
        'AMT_CREDIT',
        'AMT_ANNUITY',
        'AMT_GOODS_PRICE'
    ]
    elements_info_cols = [
        'FLAG_DOCUMENT_2',
        'FLAG_DOCUMENT_3',
        'FLAG_DOCUMENT_4',
        'FLAG_DOCUMENT_5',
        'FLAG_DOCUMENT_6',
        'FLAG_DOCUMENT_7',
        'FLAG_DOCUMENT_8',
        'FLAG_DOCUMENT_9',
        'FLAG_DOCUMENT_10',
        'FLAG_DOCUMENT_11',
        'FLAG_DOCUMENT_12',
        'FLAG_DOCUMENT_13',
        'FLAG_DOCUMENT_14',
        'FLAG_DOCUMENT_15',
        'FLAG_DOCUMENT_16',
        'FLAG_DOCUMENT_17',
        'FLAG_DOCUMENT_18',
        'FLAG_DOCUMENT_19',
        'FLAG_DOCUMENT_20',
        'FLAG_DOCUMENT_21'
    ]

    if ID:
        if val_button == 'information':
            cols = client_info_cols
        elif val_button == 'characteristics':
            cols = loan_info_cols
        elif val_button == 'elements':
            cols = elements_info_cols        
        file = X_test.loc[ID, cols].reset_index()
        file.columns = ['Feature', 'Value']
        file = file.to_dict('records')
    else:
        file = []

    return file
