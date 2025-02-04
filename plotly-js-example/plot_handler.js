function setupPlot(points, fig) {
    const plot = document.getElementById('plot');
    Plotly.newPlot(plot, fig.data, fig.layout);
    
    plot.on('plotly_relayout', function(eventData) {
        if (eventData['xaxis.range']) {
            const [start, end] = eventData['xaxis.range'].map(d => new Date(d));
            
            const filtered = points.filter(p => {
                const pointTime = Date.parse(p.time);
                return pointTime >= start.getTime() && pointTime <= end.getTime();
            });
            
            Plotly.restyle(plot, {
                'x': [filtered.map(p => p.x)],
                'y': [filtered.map(p => p.y)],
                'customdata': [filtered.map(p => [p.time])]
            }, [0]);
        }
    });
} 
