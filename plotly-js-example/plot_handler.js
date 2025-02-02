function setupPlot(points, fig) {
    var plot = document.getElementById('plot');
    Plotly.newPlot(plot, fig.data, fig.layout);
    
    plot.on('plotly_relayout', function(eventData) {
        if ('xaxis2.range' in eventData) {
            let start, end;
            [start, end] = eventData['xaxis2.range'];
            
            var filtered = points.filter(function(p) {
                var t = new Date(p.time);
                return t >= new Date(start) && t <= new Date(end);
            });
            
            Plotly.restyle(plot, {
                x: [filtered.map(p => p.x)],
                y: [filtered.map(p => p.y)],
                customdata: [filtered.map(p => [p.time])]
            }, [0]);
        }
    });
} 
