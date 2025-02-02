import plotly.graph_objects as go
import numpy as np
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta
import json

# Generate data
times = np.linspace(0, 2*np.pi, 100)
points = []
start_time = datetime(2024, 1, 25, 12)

for i, time in enumerate(times):
    points.append({
        'x': np.sin(time),
        'y': np.sin(2*time),
        'time': (start_time + timedelta(hours=i)).isoformat()
    })

# Create figure
fig = go.Figure()

# Add main scatter plot
fig.add_trace(
    go.Scatter(
        x=[p['x'] for p in points],
        y=[p['y'] for p in points],
        mode='markers',
        name='points',
        customdata=[[p['time']] for p in points],
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>time: %{customdata[0]}'
    )
)

# Add time series for range selection
fig.add_trace(
    go.Scatter(
        x=[p['time'] for p in points],
        y=[0] * len(points),
        mode='markers',
        showlegend=False,
        xaxis='x2',
        yaxis='y2'
    )
)

# Update layout
fig.update_layout(
    title='Lissajous Figure (1:2)',
    xaxis=dict(
        title='sin(t)',
        range=[-1.5, 1.5],
        domain=[0, 1],
        fixedrange=True
    ),
    yaxis=dict(
        title='sin(2t)',
        range=[-1.5, 1.5],
        domain=[0.3, 1],
        fixedrange=True
    ),
    xaxis2=dict(
        title='Time',
        rangeslider=dict(visible=True),
        type='date',
        domain=[0, 1],
        anchor='y2'
    ),
    yaxis2=dict(
        domain=[0, 0.2],
        anchor='x2',
        visible=False
    )
)

js_points = json.dumps(points)
js_fig = fig.to_json()

# Add JavaScript for filtering points
with open('simple_plot.html', 'w') as f:
    html_template = f'''
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="plot_handler.js"></script>
    </head>
    <body>
        <div id="plot"></div>
        <script>
            var points = {js_points};
            var fig = {js_fig};
            setupPlot(points, fig);
        </script>
    </body>
    </html>
    '''
    f.write(html_template)

webbrowser.open(Path('simple_plot.html').absolute().as_uri())
