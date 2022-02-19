from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from app import app
from layouts import loan_score, file_details
import callbacks

from utils.functions import *


IDS = reduce_memory_usage(
    pd.read_csv(
        'data/fulfilled_data.csv',
        usecols=['SK_ID_CURR']
    ).tail(100)
)


SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,    
    'width': 'auto',
    'padding': '2rem 1rem'
}
CONTENT_STYLE = {
    'margin-left': '18rem',
    'margin-right': '2rem',
    'padding': '2rem 1rem',
}


sidebar = html.Div(
    [
        html.H1('Loanly & Risky', style={'font-style': 'italic'}),
        html.Hr(),
        html.H6('Loan request evaluation'),
        html.Br(),
        html.Div(
            [
                dbc.Label('Customer ID:'),
                dcc.Dropdown(
                    id='sk-id-curr',
                    options=[
                        {'label': ID, 'value': ID} for ID in IDS['SK_ID_CURR']
                    ],
                    style={'color': 'black'}
                )
            ],
            className='mb-3'
        ),
        dbc.Nav(
            [
                dbc.NavLink(
                    'Loan score',
                    href='/',
                    active='exact'
                ),                
                dbc.NavLink(
                    'File details',
                    href='/file_details',
                    active='exact'
                )
            ],
            class_name='card-header-pills',
            pills=True,
            vertical=True
        ),
        html.Img(
            src='assets/logo.png',
            className='mt-5'
        )
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id='page-content', style=CONTENT_STYLE)

app.layout = html.Div(
    dbc.Container(
        [
            dcc.Location(id='url'),
            sidebar,
            content
        ]
    )
)


@ app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def render_page_content(pathname):
    if pathname in ['/', '/loan_score']:
        return loan_score    
    elif pathname == '/file_details':
        return file_details
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        dbc.Container(
            [
                html.H1('404: Not found', className='text-danger'),
                html.Hr(),
                html.P(f'The pathname {pathname} was not recognised...')
            ],
            fluid=True,
            class_name='py-3',
        ),
        className='p-3 bg-light rounded-3',
    )
