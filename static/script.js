function predictText() {
    let text = document.getElementById("inputText").value;

    if (!text) {
        alert("Please enter some text.");
        return;
    }

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("result").innerHTML = `<span style="color: red;">Error: ${data.error}</span>`;
        } else {
            let color = data.prediction === "AI-generated" ? "red" : "green";
            document.getElementById("result").innerHTML = `
                <p style="color: ${color};">
                    Prediction: <strong>${data.prediction}</strong><br>
                    Probability: ${data.probability * 100}%
                </p>`;
        }
    })
    .catch(error => console.error("Error:", error));
}
