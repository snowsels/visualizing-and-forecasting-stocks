import dash
from dash import dcc
from dash import html
from datetime import datetime as dt
import yfinance as yf
import pandas as pd
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import plotly.graph_objects as go

def get_stock_price_fig(df):
    fig=go.Figure()
    fig.add_trace(go.Scatter(mode="lines",x=df["Date"], y=df["Close"]))
    return fig
def get_sg(df, label):
    non_main=1-df.values[0]
    labels=["main", label]
    values=[non_main, df.values[0]]
    fig=go.Figure(data=[go.Pie(labels=labels, values=values,hole=0.499)])
    return fig



app = dash.Dash()


app.layout = html.Div([
    html.Div([
        html.P("Choose a ticker to start"),
         dcc.Dropdown("dropdown_tickers", options=[
         {"label": "Apple", "value": "AAPL"},
         {"label": "Tesla", "value": "TSLA"},
         {"label": "Meta", "value": "META"}],  style={"width": "25vw","height": "5vh",
         "align-items": "center",
         "padding": "0",
         "box-sizing": "0",
         "font-family": "Roboto"}),

        html.Div([
            html.Button("stock Price", className="stock-btn", id="stock"),
            html.Button("Indicators", className="indicators-btn", id="indicators"),


        ], className="buttons")
    ], className="navigation"),
html.Div([
        html.Div([
            html.P(id="ticker"),
            html.Img(id="logo")
        ], className="header"),
        html.Div(id="description", className="description_ticker"),
        html.Div([
            html.Div([], id="graphs-contents")
        ], id="main-content")

    ], className="content")

], className="container")
@app.callback(
 [Output("description", "children"), Output("logo", "src"), Output("ticker", "children")],
 [Input("dropdown_tickers", "value")]
)
def update_data(v):
 if v==None:
  raise PreventUpdate
 ticker = yf.Ticker(v)
 inf = ticker.info
 df = pd.DataFrame().from_dict(inf, orient="index").T
 df = df[["logo_url", "shortName", "longBusinessSummary"]]
 return df["longBusinessSummary"].values[0], df["logo_url"].values[0], df["shortName"].values[0]

@app.callback(
    [Output("graphs-contents", "children")],
    [Input("stock", "n_clicks"), Input("dropdown_tickers", "value")]
)
def stock_price(v, v2):
    if v== None:
        raise PreventUpdate
    df=yf.download(v2)
    df.reset_index( inplace=True)
    fig=get_stock_price_fig(df)

    return[dcc.Graph(figure=fig)]
@app.callback(
    [Output("main-content", "children"), Output("stock", "n_clicks")],
    [Input("indicators", "n_clicks"), Input("dropdown_tickers", "value")]
)
def indicators(v, v2):
 if v==None:
     raise PreventUpdate
 ticker=yf.Ticker(v2)
 df_info = pd.DataFrame.from_dict(ticker.info, orient="index").T
 df_info= df_info[["priceToBook", "profitMargins", "bookValue", "enterpriseToEbitda", "shortRatio", "beta","payoutRatio", "trailingEps"]]
 kpi_data =html.Div([
     html.Div([
     html.Div([
         html.H4("Price to Book"),
         html.P(df_info["priceToBook"])
     ]),
     html.Div([
         html.H4("Enterprise to Ebitda"),
         html.P(df_info["enterpriseToEbitda"])
     ]),
     html.Div([
         html.H4("Beta"),
         html.P(df_info["beta"])
     ]),
 ], className="kpi"),
         html.Div([
             dcc.Graph(figure=get_sg(df_info["profitMargins"], "Margins")),
             dcc.Graph(figure=get_sg(df_info["payoutRatio"], "Payout"))
         ], className="sg")
])
 return[html.Div([kpi_data], id="graphs-contents")], None

app.run_server(debug=True)