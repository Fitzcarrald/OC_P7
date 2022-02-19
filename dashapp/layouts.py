from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc


loan_score = html.Div(
    [
        dbc.Row(
            [
                html.Div(id='title'),                
                html.Div(id='decision')
            ],
            className='mb-2'
        ),
        dbc.Row(
            dcc.Graph(id='gauge-score')
        ),
        dbc.Row(
            [
                dcc.Graph(id='impact_pos'),
                dcc.Graph(id='impact_neg')
            ]
        )
    ]
)

file_details = html.Div(
    [
        dbc.Row(
            [
                html.H3('File details'),
                # html.Div(id='decision')
            ],
            className='mb-2'
        ),
        dcc.RadioItems(
            id='buttons',
            options=[
                {'label': 'Client data', 'value': 'information'},
                {'label': 'Characteristics of loan', 'value': 'characteristics'},
                {'label': 'File elements', 'value': 'elements'}
            ],
            value='information',
            inline=True,
            inputStyle={'margin-left': '50px'},
            style={'textAlign': 'right'},
            className='mb-3'
        ),
        dash_table.DataTable(
            id='table',
            columns=[{'id': i, 'name': i} for i in ['Feature', 'Value']],
            style_cell={'color': 'black', 'width': 'auto'}
        )
    ]
)
