document.getElementById("hairForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const data = {
        age: document.getElementById("age").value,
        genetics: document.getElementById("genetics").value,
        stress: document.getElementById("stress").value,
        nutritional_deficiencies: document.getElementById("nutrition").value,
        poor_hair_care_habits: document.getElementById("habits").value
    };

    try {
        const response = await fetch("https://haircareai.onrender.com/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error("API error");
        }

        const result = await response.json();

        document.getElementById("result").innerHTML = `
            <strong>Hair Fall Risk:</strong> ${result.hair_fall_risk}<br>
            <strong>Confidence:</strong> ${result.confidence}<br>
            <strong>Why:</strong> ${result.why}<br><br>
            <strong>Nutrition:</strong> ${result.nutrition_suggestions.join(", ")}<br>
            <strong>Products:</strong> ${result.product_suggestions.join(", ")}
        `;
    } catch (err) {
        document.getElementById("result").innerText = "API connection failed";
        console.error(err);
    }
});
