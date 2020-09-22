import plotly.graph_objs as go
from matplotlib import pyplot
from plotly.offline import init_notebook_mode, iplot


def get_summer(dates):
    draw = {}
    summer = [('2015-12-21', '2016-03-20'), ('2016-12-21', '2017-03-20')]
    i = 0
    for start, end in summer:
        ss = dates[(dates > start) & (dates < end)]
        if ss.shape[0] > 0:
            draw[i] = (i, ss.index[0], ss.index[-1])
            i += 1
    return draw


def plot_results(df):
    trace_high = go.Scatter(
        x=df.time,
        y=df['real'],
        name="Real",
        line=dict(color='#17BECF'),
        opacity=0.8)

    trace_low = go.Scatter(
        x=df.time,
        y=df['predict'],
        name="Predict",
        line=dict(color='black'),
        opacity=0.8)

    data = [trace_high, trace_low]

    layout = go.Layout(
        autosize=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            x=0.8,
            y=0.8,
            traceorder='normal',
            font=dict(
                size=12, ),
        ),
        width=800,
        height=500,
        yaxis=go.layout.YAxis(
            title='Precipitation amount(mm) in 24 hours'
        )
    )

    fig = go.Figure(data=data, layout=layout)

    for _, start, end in get_summer(df.time).values():
        fig.add_shape(
            type="rect",
            x0=df.time[start],
            y0=-10.0,
            x1=df.time[end],
            y1=130.0,
            line=dict(
                color="RoyalBlue",
            ),
        )
        fig.update_shapes(dict(xref='x', yref='y'))

    fig.show()


import statsmodels.api as sm


def plot_scatter(df):
    line = sm.OLS(df['real'], sm.add_constant(df['real'])).fit().fittedvalues
    trace1 = {
        "mode": "markers",
        "name": "Regression Plot",
        "type": "scatter",
        "x": df['real'],
        "y": df['predict'],
    }

    data = [trace1, ]
    layout = go.Layout({
        "title": "Prediction vs Real",
        "xaxis": {"title": "Real values", "range": [0, 120]},
        "yaxis": {"title": "Predicted values", "range": [0, 120]},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "width": 800,
        "height": 500
    })

    fig = go.Figure(data=data, layout=layout)
    fig.add_trace(go.Scatter(name='line of best fit', x=df['real'], y=line, mode='lines'))
    fig.show()