import plotly.graph_objects as go
import numpy as np
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta
import json

# Generate data
t = np.linspace(0, 2*np.pi, 100)
points = []
start_time = datetime(2024, 1, 25, 12)

for i, t_val in enumerate(t):
    points.append({
        'x': float(np.sin(t_val)),  # Convert to float for JSON
        'y': float(np.sin(2*t_val)),
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

# Add JavaScript for filtering points
with open('simple_plot.html', 'w') as f:
    html_template = '''
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div id="plot"></div>
        <script>
            var points = %s;
            var fig = %s;
            
            var plot = document.getElementById('plot');
            Plotly.newPlot(plot, fig.data, fig.layout);
            
            plot.on('plotly_relayout', function(e) {
                if ('xaxis2.range[0]' in e || 'xaxis2.range[1]' in e || 'xaxis2.range' in e) {
                    var range = [
                        e['xaxis2.range[0]'] || e['xaxis2.range'] && e['xaxis2.range'][0] || fig.layout.xaxis2.range[0],
                        e['xaxis2.range[1]'] || e['xaxis2.range'] && e['xaxis2.range'][1] || fig.layout.xaxis2.range[1]
                    ];
                    
                    var filtered = points.filter(function(p) {
                        var t = new Date(p.time);
                        return t >= new Date(range[0]) && t <= new Date(range[1]);
                    });
                    
                    Plotly.restyle(plot, {
                        x: [filtered.map(p => p.x)],
                        y: [filtered.map(p => p.y)],
                        customdata: [filtered.map(p => [p.time])]
                    }, [0]);
                }
            });
        </script>
    </body>
    </html>
    ''' % (json.dumps(points), fig.to_json())
    f.write(html_template)

webbrowser.open(Path('simple_plot.html').absolute().as_uri()) 
