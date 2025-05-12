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
        return data;
    } catch (error) {
        console.error("Fetch Error:", error);
        return text;
    }
}

window.fetchCheckedAndCorrectedText = async (text) => {
    try {
        let response = await fetch("https://localhost:5001/check", {
            method: "POST",
            mode: "cors",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        let data = await response.json();
        return data;
    } catch (error) {
        console.error("Fetch Error:", error);
        return text;
    }
}

window.fetchActiveLearning = async (original, suggestions, chosen) => {
    try {
        let response = await fetch("https://localhost:5001/feedback", {
            method: "POST",
            mode: "cors",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ original, suggestions, chosen })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
    } catch (error) {
        console.error("Fetch Error:", error);
    }
}

window.fetchSynonyms = async (word) => {
    try {
        let response = await fetch("https://localhost:5001/synonym", {
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