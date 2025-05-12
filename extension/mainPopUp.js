function addOption(sel, recommendation, activeLearning = false) {
    const container = document.getElementById("recommendationsContainer");

    const item = document.createElement("div");
    Object.assign(item.style, {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "4px 8px",
        margin: "2px 0",
        cursor: "pointer",
        borderRadius: "4px",
        backgroundColor: "#333",
        transition: "background-color 0.2s",
    });

    item.addEventListener("mouseenter", () => {
        item.style.backgroundColor = "#444";
    });
    item.addEventListener("mouseleave", () => {
        item.style.backgroundColor = "#333";
    });

    // Left: the text
    const span = document.createElement("span");
    span.textContent = recommendation;
    span.style.flex = "1";

    // Right: the checkmark
    const check = document.createElement("img");
    check.src = chrome.runtime.getURL("media/checkmark.svg");
    Object.assign(check.style, {
        width: "16px",
        height: "16px",
        marginLeft: "8px",
        opacity: "0",
        transition: "opacity 0.2s",
    });

    // Fade checkmark in on hover/focus
    item.addEventListener("mouseenter", () => { check.style.opacity = "1"; });
    item.addEventListener("mouseleave", () => { check.style.opacity = "0"; });
    item.addEventListener("focus", () => { check.style.opacity = "1"; });
    item.addEventListener("blur", () => { check.style.opacity = "0"; });

    // Clicking the entire item applies the recommendation
    item.tabIndex = 0;  // make it focusable
    item.addEventListener("click", () => {
        setFieldValue(recommendation, sel.start, sel.end);
        if (activeLearning) {
            // TODO
        }
        document.getElementById("wordSuggestPopup")?.remove();
    });
    item.addEventListener("keydown", e => {
        if (e.key === "Enter" || e.key === " ") {
            setFieldValue(recommendation, sel.start, sel.end);
            if (activeLearning) {
                // TODO
            }
            document.getElementById("wordSuggestPopup")?.remove();
            e.preventDefault();
        }
    });

    item.append(span, check);
    container.appendChild(item);
}

function popUpContent(newSelection, popup) {
    const header = document.createElement("h3");
    header.textContent = "Text corrections using AI";
    Object.assign(header.style, {
        margin: "0 0 8px 0",
        fontSize: "16px",
        textAlign: "center",
        color: "white",
    });
    popup.appendChild(header);

    // Button container
    const container = document.createElement("div");
    Object.assign(container.style, {
        display: "flex",
        gap: "8px",
        justifyContent: "center",
        marginBottom: "5px",
    });

    // ACTION ON BUTTON CLICK
    const onLevenshtein = async sel => {
        const recommendations = await window.fetchCorrectedWord(sel.text);
        const container = document.getElementById("recommendationsContainer");
        container.innerHTML = "";
        recommendations.forEach(rec => addOption(sel, rec, false));
    };

    const onCorrect = async sel => {
        const recommendations = await window.fetchCorrectedText(sel.text);
        const container = document.getElementById("recommendationsContainer");
        container.innerHTML = "";
        recommendations.forEach(rec => addOption(sel, rec, true));
    };

    // Create buttons
    const isWord = /^[\p{L}]+$/u.test(newSelection.text);

    const btnLev = document.createElement("button");
    btnLev.textContent = "Levenshtein";
    btnLev.disabled = !isWord;
    Object.assign(btnLev.style, {
        flex: "1",
        padding: "6px 0",
        border: "1px solid #555",
        background: isWord ? "#333" : "#222",
        color: isWord ? "white" : "#777",
        borderRadius: "4px",
        cursor: isWord ? "pointer" : "not-allowed",
    });
    btnLev.addEventListener("click", () => {
        onLevenshtein(newSelection);
    });

    const btnCor = document.createElement("button");
    btnCor.textContent = "Correct";
    Object.assign(btnCor.style, {
        flex: "1",
        padding: "6px 0",
        border: "1px solid #555",
        background: "#333",
        color: "white",
        borderRadius: "4px",
        cursor: "pointer",
    });
    btnCor.addEventListener("click", () => {
        onCorrect(newSelection);
    });

    container.append(btnLev, btnCor);
    popup.appendChild(container);

    const recommendationsDiv = document.createElement("div");
    recommendationsDiv.id = "recommendationsContainer";
    recommendationsDiv.textContent = "No recommendations available for the selected text";
    Object.assign(recommendationsDiv.style, {
        display: "flex",
        flexDirection: "column",
        margin: "0 0 8px 0",
        fontSize: "16px",
        textAlign: "center",
    });
    popup.appendChild(recommendationsDiv);

    // Default action: click first enabled button if user does nothing
    window.requestAnimationFrame(() => {
        if (!btnLev.disabled) {
            btnLev.click();
        } else {
            btnCor.click();
        }
    });
};

window.showMainPopUp = (newSelection, clickX, clickY) => {
    document.getElementById("wordSuggestPopup")?.remove();

    const popup = document.createElement("div");
    popup.id = "wordSuggestPopup";
    Object.assign(popup.style, {
        position: "absolute",
        display: "flex",
        flexDirection: "column",
        padding: "1em",
        border: "1px solid #888",
        borderRadius: "4px",
        background: "#242424",
        color: "white",
        boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
        zIndex: 10000,
    });
    document.body.appendChild(popup);

    // (re-)position the popup around clickX/clickY
    const positionPopup = () => {
        const pw = popup.offsetWidth, ph = popup.offsetHeight;
        const sx = window.scrollX, sy = window.scrollY;
        const vw = window.innerWidth, vh = window.innerHeight;

        let left = clickX - pw / 2;
        left = Math.max(sx + 5, Math.min(left, sx + vw - pw - 5));

        let top = clickY + 5;
        if (top + ph > sy + vh) {
            top = clickY - 5 - ph;
            top = Math.max(sy + 5, top);
        }

        popup.style.left = `${left}px`;
        popup.style.top = `${top}px`;
    };

    popUpContent(newSelection, popup);

    positionPopup();

    // once recommendations appear, reposition again
    const recContainer = popup.querySelector("#recommendationsContainer");
    if (recContainer) {
        const mo = new MutationObserver(() => {
            positionPopup();
        });
        // watch for children being added
        mo.observe(recContainer, { childList: true });
        // mo.disconnect();
    }

    // outside‐click dismissal (2h lost here)
    setTimeout(() => {
        function outsideClickListener(e) {
            if (!e.target.closest("#wordSuggestPopup")) {
                popup.remove();
                document.removeEventListener("click", outsideClickListener);
            }
        }
        document.addEventListener("click", outsideClickListener);
    }, 0);

    popup.addEventListener("click", e => e.stopPropagation());
};