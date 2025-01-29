from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Generate data
start_time = datetime(2024, 1, 25, 12, 0)
t = np.linspace(0, 2*np.pi, 100)
points = [
    {
        'x': np.sin(t_val),
        'y': np.sin(2*t_val),
        'time': start_time + timedelta(hours=i)
    }
    for i, t_val in enumerate(t)
]

# Create Dash app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    dcc.Graph(id='lissajous-plot'),
    dcc.RangeSlider(
        id='time-slider',
        min=0,
        max=len(points)-1,
        value=[0, len(points)-1],
        marks={
            0: points[0]['time'].strftime('%Y-%m-%d %H:%M'),
            len(points)-1: points[-1]['time'].strftime('%Y-%m-%d %H:%M')
        },
        step=1  # Force integer steps
    )
])

@app.callback(
    Output('lissajous-plot', 'figure'),
    Input('time-slider', 'value')
)
def update_figure(time_range):
    # Convert float indices to integers
    start_idx = int(time_range[0])
    end_idx = int(time_range[1]) + 1
    
    # Filter points by time range
    filtered_points = points[start_idx:end_idx]
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=[p['x'] for p in filtered_points],
            y=[p['y'] for p in filtered_points],
            mode='markers',
            name='points',
            customdata=[[p['time'].strftime('%Y-%m-%d %H:%M')] for p in filtered_points],
            hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>time: %{customdata[0]}<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Lissajous Figure (1:2)',
        xaxis=dict(range=[-1.5, 1.5], fixedrange=True),
        yaxis=dict(range=[-1.5, 1.5], fixedrange=True)
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True) 
