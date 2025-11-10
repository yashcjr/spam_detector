document.getElementById('predictBtn').addEventListener('click', async () => {
    const userInput = document.getElementById('userInput').value.trim();
    const resultDiv = document.getElementById('result');

    if (!userInput) {
        resultDiv.textContent = "⚠️ Please enter a message or URL!";
        resultDiv.className = "result spam";
        resultDiv.classList.remove("hidden");
        return;
    }

    resultDiv.textContent = "⏳ Predicting...";
    resultDiv.className = "result hidden";
    resultDiv.classList.remove("hidden");

    try {
       /*  const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: userInput })
        }); */
           const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: userInput })
        });



        const data = await response.json();
        const prediction = data.result.trim();

        resultDiv.textContent = prediction;
        resultDiv.classList.remove("hidden");

        if (prediction.toLowerCase().includes("spam")) {
            resultDiv.className = "result spam";
        } else {
            resultDiv.className = "result ham";
        }

    } catch (err) {
        resultDiv.textContent = "❌ Error predicting spam. Check your server!";
        resultDiv.className = "result spam";
        resultDiv.classList.remove("hidden");
        console.error("Prediction error:", err);
    }
});
