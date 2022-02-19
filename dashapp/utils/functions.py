import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px


# Reduce the dataframe numeric memory
def reduce_memory_usage(df):    
    for col in df.select_dtypes('integer').columns:
        col_min = df[col].min()
        col_max = df[col].max()

        if str(df[col].dtype) in ['int16', 'int32', 'int64']:
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
    
    for col in df.select_dtypes('floating').columns:
        col_min = df[col].min()
        col_max = df[col].max()

        if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
            df[col] = df[col].astype(np.float16)
        elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
            df[col] = df[col].astype(np.float32)

    return df

# Predict the loan's class and calculate its probability
def binary_prediction(ID, model, X_test):
    defaulter_class = model.predict(X_test.loc[[ID]])[0].astype('int')
    prob = model.predict_proba(X_test.loc[[ID]])[0][1]
    return defaulter_class, prob

# Plots
def gauge_chart(prob):
    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode='number+gauge',
            value=100 * prob,
            number={'suffix': '%'},
            gauge={
                'bar': {'color': 'black'},
                'axis': {'range': [0, 100]},
                'threshold': {'value': 60},
                'steps': [
                    {'range': [0, 30], 'color': 'green'},
                    {'range': [30, 60], 'color': 'yellow'},
                    {'range': [60, 100], 'color': 'red'}
                ],
            },
            title="Client's default probability",
            domain={'row': 0, 'column': 0}
        )
    )
    fig.add_trace(
        go.Indicator(
            mode='number+delta',
            value=np.round(prob / (1 - prob), 2),
            delta={'reference': 1, 'increasing.color': 'red',
                   'decreasing.color': 'green'},
            title="Loan's odds ratio:",
            domain={'row': 0, 'column': 1}
        )
    )
    fig.update_layout(
        grid = {'rows': 1, 'columns': 2, 'pattern': 'independent'}
        # grid={'rows': 1, 'columns': 2}
    )

    return fig


def feature_impact(ID, X_test_prep, feature_coef):
    vector = X_test_prep.loc[ID] if ID else pd.Series(10 * [0])
    feature_imp = np.exp(feature_coef * vector)
    feature_imp_pos = feature_imp[feature_imp.gt(
        1)].nlargest(10).apply(np.round, decimals=2).rename('impact')
    feature_imp_neg = feature_imp[feature_imp.lt(
        1)].nsmallest(10).apply(np.round, decimals=2).rename('impact')

    fig_pos = px.bar(
        feature_imp_pos,
        x='impact',
        color_discrete_sequence=['green'],
        orientation='h',
        text='impact',
        title='Top 10 elements in favor'
    )
    fig_neg = px.bar(
        feature_imp_neg,
        x='impact',
        color_discrete_sequence=['red'],
        orientation='h',
        text='impact',
        title='Top 10 elements against',
    )

    if ID:
        fig_pos.update_yaxes(categoryorder='total ascending')
        fig_neg.update_yaxes(categoryorder='total descending')
    else:
        fig_pos.update_xaxes(rangemode='tozero').update_yaxes(
            rangemode='tozero')
        fig_neg.update_xaxes(rangemode='tozero').update_yaxes(
            rangemode='tozero')

    return fig_pos, fig_neg
