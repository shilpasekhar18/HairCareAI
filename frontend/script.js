document.getElementById("hairForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const data = {
    age: document.getElementById("age").value,
    genetics: document.getElementById("genetics").value,
    stress: document.getElementById("stress").value,
    nutritional_deficiencies: document.getElementById("nutrition").value,
    poor_hair_care_habits: document.getElementById("habits").value
  };

  try {
    const res = await fetch("http://127.0.0.1:5001/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    const result = await res.json();

    document.getElementById("result").innerHTML = `
      <h3>Hair Fall Risk: ${result.hair_fall_risk}</h3>
      <p><b>Confidence:</b> ${result.confidence}</p>
      <p><b>Why:</b> ${result.why}</p>
      <p><b>Nutrition:</b> ${result.nutrition_suggestions.join(", ")}</p>
      <p><b>Products:</b> ${result.product_suggestions.join(", ")}</p>
    `;
  } catch (err) {
    document.getElementById("result").innerText = "API connection failed";
    console.error(err);
  }
});
