window.fetchCorrectedText = async (text) => {
    try {
        let response = await fetch("https://localhost:5001/correct", {
            method: "POST",
            mode: "cors",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        let data = await response.json();
        return data.corrected;
    } catch (error) {
        console.error("Fetch Error:", error);
        return text;
    }
}

window.fetchCorrectedWord = async (word) => {
    try {
        let response = await fetch("https://localhost:5001/word", {
            method: "POST",
            mode: "cors",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ word: word })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        let data = await response.json();
        return data.suggestions;
    } catch (error) {
        console.error("Fetch Error:", error);
        return word;
    }
}