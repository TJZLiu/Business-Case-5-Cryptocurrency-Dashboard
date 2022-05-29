import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from statsmodels.tsa.api import ExponentialSmoothing, Holt
import statsmodels.api as sm
import datetime as DT

from stockstats import StockDataFrame
######################################################Built-in functions################################################

def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

def Yestoday_Price(df):
    YP = df.iloc[-1].to_frame()
    YP.columns = [str(get_df_name(df))]
    return YP

def get_date_followweek(df):
    list = []
    for i in range(1, 8):
        list.append(df.index[-1] + DT.timedelta(days=i))
    return list
######################################################Data##############################################################

ADA_USD = yf.download("ADA-USD")
ATOM_USD = yf.download("ATOM-USD")
AVAX_USD = yf.download("AVAX-USD")
AXS_USD = yf.download("AXS-USD")
BTC_USD = yf.download("BTC-USD")
ETH_USD = yf.download("ETH-USD")
LINK_USD = yf.download("LINK-USD")
LUNA1_USD = yf.download("LUNA1-USD")
MATIC_USD = yf.download("MATIC-USD")
SOL_USD = yf.download("SOL-USD")

Cryptocurrencies = [ADA_USD,ATOM_USD,AVAX_USD,AXS_USD,BTC_USD,ETH_USD,LINK_USD,LUNA1_USD,MATIC_USD,SOL_USD]

Cryptocurrencies_factors = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

Time_period_factors = ['1 Day','3 Days','1 Week','2 Weeks','Month']

Yestoday_Price_Table = pd.concat([Yestoday_Price(ADA_USD),Yestoday_Price(ATOM_USD),Yestoday_Price(AVAX_USD),
                                  Yestoday_Price(AXS_USD),Yestoday_Price(BTC_USD),Yestoday_Price(ETH_USD),
                                  Yestoday_Price(LINK_USD),Yestoday_Price(LUNA1_USD),Yestoday_Price(MATIC_USD),
                                  Yestoday_Price(SOL_USD)], axis=1, join='inner').T

######################################################Interactive Components############################################

Cryptocurrencies_options = [dict(label=get_df_name(Cryptocurrency),
                                 value=get_df_name(Cryptocurrency)) for Cryptocurrency in Cryptocurrencies]

Cryptocurrencies_factors_options = [dict(label=factors, value=factors) for factors in Cryptocurrencies_factors]

Time_period_factors_options = [dict(label=factors, value=factors) for factors in Time_period_factors]

Selector_Cryptocurrency = dcc.Dropdown(
        id='Cryptocurrency_drop',
        options=Cryptocurrencies_options,
        value='BTC_USD'
    )

Selector_factors = dcc.RadioItems(
        id='dropdown_factor',
        options=Cryptocurrencies_factors_options,
        value='Adj Close'
    )

Selector_timeperiod = dcc.RadioItems(
        id='time_option',
        options=Time_period_factors_options,
        value='1 Day',
        inline=True
    )

dropdown_scope = dcc.RadioItems(
        id='scopes_option',
        options=[{'label': 'World', 'value': 'world'},
                {'label': 'Europe', 'value': 'europe'},
                {'label': 'Asia', 'value': 'asia'},
                {'label': 'Africa', 'value': 'africa'},
                {'label': 'North america', 'value': 'north america'},
                {'label': 'South america', 'value': 'south america'}],
        value='world',
    )

Selector_indicators = dcc.Dropdown(
        id='indicator_option',
        options=[{'label': 'SMA', 'value': 'close_5_sma'},
                {'label': 'EMA', 'value': 'close_5_ema'},
                {'label': 'MSTD', 'value': 'close_5_mstd'},
                {'label': 'MVAR', 'value': 'close_5_mvar'},
                {'label': 'Bolling', 'value': 'boll'}],###
        value=None,
        placeholder="Select an indicator",
    )

##################################################APP###################################################################

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1( "Crypto Dashboard")
        ],className='container', id='head')
    ],className='title'),
####first column
    html.Div([
        html.Div([
            html.Br(),
            html.Div([
                html.Div([
                    html.Img(id='cryptologo',height=200),
                    html.H2(id='shortName'),
                ],className='container'),
            ],className='card'),

            html.Br(),
            html.Div([
                html.Div([
                    html.Label(id='exchangeTimezoneName'),
                    html.Br(),
                    html.Label(id='previousClose'),
                    html.Br(),
                    html.Label(id='regularMarketOpen'),
                    html.Br(),
                    html.Label(id='twoHundredDayAverage'),
                    html.Br(),
                    html.Label(id='volume24Hr'),
                    html.Br(),
                    html.Label(id='averageDailyVolume10Day'),
                    html.Br(),
                    html.Label(id='regularMarketVolume'),
                    html.Br(),
                    html.Label(id='dayLow'),
                    html.Br(),
                    html.Label(id='dayHigh'),
                ],className='container')
            ],className='card'),

            html.Br(),
            html.Div([
                html.Div([
                    html.Label(id='description')
                ],className='container')
            ],className='card'),

            html.Br(),
            html.Div([
                html.Div([
                    html.Img(src=app.get_asset_url('novaims_logo.png'),height=200),
                    html.H4("Authors"),
                    dcc.Markdown("""\
                      Laura Isabella Cuna (m20211312@novaims.unl.pt)    
                      Amelie Florentine Langenstein (m20210637@novaims.unl.pt)  
                      Tongjiuzhou Liu (m20211012@novaims.unl.pt)  
                      Nina Urbancic (m20211314@novaims.unl.pt)  
                    """, style={"text-align": "center", "font-size": "16px"}),
                    html.H4("Project Info"),
                    dcc.Markdown("""\
                     COURSE : 
                     Business Case with Data Science    
                     BC5 : 
                     Crypto Dashboard
                    """, style={"text-align": "center", "font-size": "16px"}),
                    html.H4("Instructors"),
                    dcc.Markdown("""
                    FERNANDO LUCAS BAÇÃO
                    JOÃO PEDRO FONSECA
                    HUGO SAISSE MENTZINGEN DA SILVA""", style={"text-align": "center", "font-size": "16px"}),
                ],className='container'),
            ],className='card')
        ], style={'width': '16%'}),
####second column
        html.Div([
            html.Br(),
            html.Div([
                html.Div([
                    html.Div([
                        Selector_Cryptocurrency,
                    ], style={'width': '30%'},className='container3'),
                    html.Div([
                        Selector_indicators,
                    ], style={'width': '30%'}, className='container3'),
                    html.Div([
                        Selector_timeperiod,
                    ], style={'width': '30%'},className='container3'),
                ], style={'display': 'flex'}),
            ],className='cardcenter'),
            ####main plot
            html.Br(),
            html.Div([
                html.Div([
                    dcc.Graph(id='main_plot')],className='container'),
            ],className='cardcenter'),
            ####3 predictions
            html.Br(),
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Holt Linear Prediction"),
                        dcc.Graph(id='holt_linear_plot')
                    ],className='container'),
                ], style={'width': '30%'},className='card'),
                html.Div([
                    html.Div([
                        html.H3("Holt Winters Prediction"),
                        dcc.Graph(id='holt_winters_plot')
                    ],className='container'),
                ], style={'width': '30%'},className='card'),
                html.Div([
                    html.Div([
                        html.H3("Seasonal ARIMA Prediction"),
                        dcc.Graph(id='sarima_plot')
                    ],className='container')
                ], style={'width': '30%'},className='card')
            ], style={'display': 'flex'},className='card3'),
            ##### the MAP
            html.Br(),
            html.Div([
                html.Div([
                    html.Div([
                        html.H2("Legalty Monitor"),
                    ], className='container'),
                    html.Div([
                        html.Div([], style={'width': '20%'}),
                        html.Div([dropdown_scope], style={'width': '60%'}),
                        html.Div([], style={'width': '20%'}),
                    ], style={'display': 'flex'}, className='container'),
                    html.Div([
                        dcc.Graph(id='choropleth')
                    ], className='container'),
                    html.Br()
                ], className='cardcenter'),
                html.Br()
            ]),
        ], style={'width': '60%'}),
####third column
        html.Div([
            html.Br(),
            html.Div([
                html.Div([
                    html.H3("Other Cryptocurrencies Yesterday Price "),
                    dash_table.DataTable(
                        id='other_currencies_price_table',style_data={'whiteSpace': 'normal', 'height': 'auto', 'lineHeight': '15px'},
                        style_cell={'overflow': 'hidden','textOverflow': 'ellipsis','maxWidth': 0, 'textAlign': 'center'},
                    ),
                ],className='container'),
            ],className='card'),
            ### twitter
            html.Br(),
            html.Div([
                html.Div([
                    html.Iframe(srcDoc='''
                            <a class="twitter-timeline" href="https://twitter.com/topnewsbitcoin/lists/bitcoin?ref_src=twsrc%5Etfw">
                            A Twitter List by topnewsbitcoin</a> 
                            <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                            ''', height=1400, width=375)
                ],className='container'),
            ],className='card'),
        ], style={'width': '24%'})
    ], style={'display': 'flex',}),
])





################################################Main Plot Callbacks#####################################################
@app.callback(
    [
        Output("main_plot", "figure"),
    ],
    [
        Input("Cryptocurrency_drop", "value"),
        Input("time_option", "value"),
        Input("indicator_option", "value")
    ]
)

################################################ Main Plot #############################################################

def candlestick_plot(Cryptocurrency_name,period,indicator):

    data = globals()[Cryptocurrency_name]

    logic = {'Open': 'first','High': 'max','Low': 'min','Close': 'last','Volume': 'sum'}

    if period == '1 Day':
        df = data.tail(120)
    elif period == '3 Days':
        df = data.resample('3D').apply(logic).tail(120)
    elif period == '1 Week':
        df = data.resample('W').apply(logic).tail(120)
        df.rename({df.index[-1]: data.index[-1]}, inplace=True)
    elif period == '2 Weeks':
        df = data.resample('2W').apply(logic).tail(120)
        df.rename({df.index[-1]: data.index[-1]}, inplace=True)
    elif period == 'Month':
        df = data.resample('M').apply(logic)
        df.rename({df.index[-1]: data.index[-1]}, inplace=True)
    else:
        df = data

    fig_candlestick = make_subplots(specs=[[{"secondary_y": True}]])

    fig_candlestick.add_trace(go.Bar(x=df.index,y=df['Volume'],name='Volume',marker_color='grey'),
                              secondary_y=False)

    fig_candlestick.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],
                                             low=df['Low'],close=df['Close'],name=Cryptocurrency_name),
                              secondary_y=True, )
    df = StockDataFrame.retype(df[["Open", "Close", "High", "Low", "Volume"]])

    if indicator == None :
        df = df
    elif indicator == 'boll':
        fig_candlestick.add_trace(go.Scatter(x=df.index, y=df['boll'], name='baseline'),
                                  secondary_y=True)
        fig_candlestick.add_trace(go.Scatter(x=df.index, y=df['boll_ub'], name='upper band'),
                                  secondary_y=True)
        fig_candlestick.add_trace(go.Scatter(x=df.index, y=df['boll_lb'], name='lower band'),
                                  secondary_y=True)
    elif indicator == 'macd':
        df = df
    else :
        fig_candlestick.add_trace(go.Scatter(x=df.index, y=df[indicator], name=indicator,marker_color='Blue'),
                                  secondary_y=True)

    fig_candlestick.update_layout(margin=dict(l=0,r=0,b=0,t=0,pad=0),
                                  plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',showlegend=False)
    fig_candlestick.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
    fig_candlestick.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')
    fig_candlestick.update_yaxes(range=[0, df['volume'].max() * 4], secondary_y=False)

    return [fig_candlestick]

############################################ Other price Table and logo ################################################
@app.callback(
    [
        Output("other_currencies_price_table", "data"),
        Output("cryptologo", "src"),
        Output("description", "children"),
        Output("shortName", "children"),
        Output("exchangeTimezoneName", "children"),
        Output("previousClose", "children"),
        Output("regularMarketOpen", "children"),
        Output("twoHundredDayAverage", "children"),
        Output("volume24Hr", "children"),
        Output("averageDailyVolume10Day", "children"),
        Output("regularMarketVolume", "children"),
        Output("dayLow", "children"),
        Output("dayHigh", "children"),
    ],
    [
        Input("Cryptocurrency_drop", "value")
    ]
)

def other_prices_table_and_logo(Cryptocurrency_name):
    #########TABLE##################################
    prices_table = Yestoday_Price_Table[Yestoday_Price_Table.index != Cryptocurrency_name]\
        .round(6).reset_index().rename(columns={'index': ''})
    prices_table = prices_table.loc[:, ~prices_table.columns.isin(['Adj Close', 'Volume'])].to_dict(orient='records')

    #########LOGO##################################
    logo_address = 'assets/' + Cryptocurrency_name.rpartition('_')[0] + '.png'


    #########INFO###################################
    the_name = Cryptocurrency_name.replace("_", "-")
    the_info = yf.Ticker(the_name)

    return prices_table, \
           str(logo_address),\
           str(the_info.info['description']),\
           str(the_info.info['shortName']),\
           'Exchange Timezone Name: ' + str(the_info.info['exchangeTimezoneName']),\
           'Previous Close Price: ' + str(the_info.info['previousClose']),\
           'Regular MarkeT Open Price: ' + str(the_info.info['regularMarketOpen']),\
           '200 days Average Price: ' + str(the_info.info['twoHundredDayAverage']),\
           'Volume 24 Hour: ' + str(the_info.info['volume24Hr']),\
           'Average Daily Volume of 10 Days: ' + str(the_info.info['averageDailyVolume10Day']),\
           'Regular Market Volume: ' + str(the_info.info['regularMarketVolume']),\
           'Day Low Price: ' + str(the_info.info['dayLow']), \
           'Day High Price: ' + str(the_info.info['dayHigh'])


################################################ Prediction plots ######################################################
@app.callback(
    [
        Output("holt_linear_plot", "figure"),
        Output("holt_winters_plot", "figure"),
        Output("sarima_plot", "figure"),
    ],
    [
        Input("Cryptocurrency_drop", "value")
    ]
)

def prediction_plots(Cryptocurrency_name):
    prediction_data = globals()[Cryptocurrency_name].tail(120)
    ########## HOLT LINEAR ########################
    Holtfit1 = Holt(np.asarray(prediction_data['Open'])).fit(smoothing_level=0.3, smoothing_trend=0.1)
    Holtfit2 = Holt(np.asarray(prediction_data['High'])).fit(smoothing_level=0.3, smoothing_trend=0.1)
    Holtfit3 = Holt(np.asarray(prediction_data['Low'])).fit(smoothing_level=0.3, smoothing_trend=0.1)
    Holtfit4 = Holt(np.asarray(prediction_data['Close'])).fit(smoothing_level=0.3, smoothing_trend=0.1)
    HOLT_LINEAR_Prediction = pd.DataFrame()
    HOLT_LINEAR_Prediction['Date'] = get_date_followweek(prediction_data)
    HOLT_LINEAR_Prediction['Open'] = Holtfit1.forecast(7)
    HOLT_LINEAR_Prediction['High'] = Holtfit2.forecast(7)
    HOLT_LINEAR_Prediction['Low'] = Holtfit3.forecast(7)
    HOLT_LINEAR_Prediction['Close'] = Holtfit4.forecast(7)
    HOLT_LINEAR_Prediction = HOLT_LINEAR_Prediction.set_index('Date')



    HOLT_LINEAR_Prediction_data=[go.Candlestick(x=HOLT_LINEAR_Prediction.index,
                                         open=HOLT_LINEAR_Prediction['Open'],
                                         high=HOLT_LINEAR_Prediction['High'],
                                         low=HOLT_LINEAR_Prediction['Low'],
                                         close=HOLT_LINEAR_Prediction['Close'])]

    layout = dict(xaxis_rangeslider_visible=False,margin=dict(l=0,r=0,b=0,t=0,pad=0),
                                  plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')

    ########## Holt-Winters ########################
    HWMfit1 = ExponentialSmoothing(np.asarray(prediction_data['Open']), seasonal_periods=7, trend='add', seasonal='add', ).fit()
    HWMfit2 = ExponentialSmoothing(np.asarray(prediction_data['High']), seasonal_periods=7, trend='add', seasonal='add', ).fit()
    HWMfit3 = ExponentialSmoothing(np.asarray(prediction_data['Low']), seasonal_periods=7, trend='add', seasonal='add', ).fit()
    HWMfit4 = ExponentialSmoothing(np.asarray(prediction_data['Close']), seasonal_periods=7, trend='add',
                                   seasonal='add', ).fit()

    HOLT_WINTERS_Prediction = pd.DataFrame()
    HOLT_WINTERS_Prediction['Date'] = get_date_followweek(prediction_data)
    HOLT_WINTERS_Prediction['Open'] = HWMfit1.forecast(7)
    HOLT_WINTERS_Prediction['High'] = HWMfit2.forecast(7)
    HOLT_WINTERS_Prediction['Low'] = HWMfit3.forecast(7)
    HOLT_WINTERS_Prediction['Close'] = HWMfit4.forecast(7)
    HOLT_WINTERS_Prediction = HOLT_WINTERS_Prediction.set_index('Date')

    HOLT_WINTERS_Prediction_data=[go.Candlestick(x=HOLT_WINTERS_Prediction.index,
                                         open=HOLT_WINTERS_Prediction['Open'],
                                         high=HOLT_WINTERS_Prediction['High'],
                                         low=HOLT_WINTERS_Prediction['Low'],
                                         close=HOLT_WINTERS_Prediction['Close'])]

    ########## SARIMA ########################
    SARIMA_Prediction = pd.DataFrame()
    SARIMA_Prediction['Date'] = get_date_followweek(prediction_data)
    SARIMA_Prediction.index = SARIMA_Prediction.Date

    ARIMAfit1 = sm.tsa.statespace.SARIMAX(prediction_data['Open'], seasonal_order=(0, 1, 1, 7)).fit()
    ARIMAfit2 = sm.tsa.statespace.SARIMAX(prediction_data['High'], seasonal_order=(0, 1, 1, 7)).fit()
    ARIMAfit3 = sm.tsa.statespace.SARIMAX(prediction_data['Low'], seasonal_order=(0, 1, 1, 7)).fit()
    ARIMAfit4 = sm.tsa.statespace.SARIMAX(prediction_data['Close'], seasonal_order=(0, 1, 1, 7)).fit()

    SARIMA_Prediction['Open'] = ARIMAfit1.predict(start=SARIMA_Prediction.Date[0], end=SARIMA_Prediction.Date[-1])
    SARIMA_Prediction['High'] = ARIMAfit2.predict(start=SARIMA_Prediction.Date[0], end=SARIMA_Prediction.Date[-1])
    SARIMA_Prediction['Low'] = ARIMAfit3.predict(start=SARIMA_Prediction.Date[0], end=SARIMA_Prediction.Date[-1])
    SARIMA_Prediction['Close'] = ARIMAfit4.predict(start=SARIMA_Prediction.Date[0], end=SARIMA_Prediction.Date[-1])

    SARIMA_Prediction_data = [go.Candlestick(x=SARIMA_Prediction.index,
                                                   open=SARIMA_Prediction['Open'],
                                                   high=SARIMA_Prediction['High'],
                                                   low=SARIMA_Prediction['Low'],
                                                   close=SARIMA_Prediction['Close'])]

    return go.Figure(data=HOLT_LINEAR_Prediction_data,layout=layout),\
           go.Figure(data=HOLT_WINTERS_Prediction_data,layout=layout),\
           go.Figure(data=SARIMA_Prediction_data,layout=layout)

################################################ Choropleth Plot #######################################################
@app.callback(
    [
        Output("choropleth", "figure"),
    ],
    [
        Input('scopes_option', 'value')
    ]
)

def plots(continent):
    df_map = pd.read_csv('Countries.csv')

    fig_map = px.choropleth(locations=df_map['Country'],
                        locationmode="ISO-3",
                        color=df_map['Legality'],
                        color_discrete_map={'Illegal': 'darkred',
                                            'Some significant concerns': 'peachpuff',
                                            'Legal': 'lightgreen'},
                        scope=continent)
    fig_map.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0),
                          legend=dict(title=None, orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"),
                                  plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')

    return [fig_map]

#######################################################################################################################

if __name__ == '__main__':
    app.run_server(debug=True)