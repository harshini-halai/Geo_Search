function initClusterChart(clusterDist) {
  const ctx = document.getElementById('clusterDoughnut').getContext('2d');
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: Object.keys(clusterDist),
      datasets: [{
        data: Object.values(clusterDist),
        backgroundColor: ['#38bdf8', '#818cf8', '#c084fc', '#f472b6', '#34d399'],
        borderWidth: 2,
        borderColor: '#0d1424'
      }]
    },
    options: { 
      responsive: true,
      maintainAspectRatio: false,
      cutout: '72%',
      plugins: { 
        legend: { display: false },
        tooltip: {
          backgroundColor: '#070b14',
          titleFont: { family: 'JetBrains Mono' },
          bodyFont: { family: 'JetBrains Mono' },
          borderColor: '#334155',
          borderWidth: 1
        }
      } 
    }
  });
}
