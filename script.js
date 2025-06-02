document.getElementById('diabetes-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    document.getElementById('loader').style.display = 'block';
    document.getElementById('result').innerText = '';

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    document.getElementById('loader').style.display = 'none';
    document.getElementById('result').innerText = `Result: ${result.result}`;

    // Update Chart
    const ctx = document.getElementById('probChart').getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Diabetic', 'Not Diabetic'],
            datasets: [{
                label: 'Probability',
                data: [result.probability, 100 - result.probability],
                backgroundColor: ['#dc3545', '#28a745']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
});
