document.querySelector("form").addEventListener("submit", async function (e) {
  e.preventDefault();

  const formData = new FormData(this);
  const data = {};
  formData.forEach((value, key) => {
    data[key] = value;
  });

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    const result = await response.json();
    const resultBox = document.getElementById("result-box");
    const resultText = document.getElementById("prediction-result");

    if (result.prediction === "Canceled") {
      resultBox.className = "result-box error";
      resultText.textContent = "⚠️ This booking is likely to be CANCELED.";
    } else {
      resultBox.className = "result-box success";
      resultText.textContent = "✅ This booking is likely to be KEPT.";
    }

    resultBox.style.display = "block";
  } catch (err) {
    alert("Something went wrong while contacting the server.");
    console.error(err);
  }
});
