document.addEventListener("focusin", function (event) {
    let target = event.target;

    // Normal input/textarea OR a custom contenteditable div
    if (
        target.tagName === "TEXTAREA" ||
        (target.tagName === "INPUT" && target.type === "text") ||
        target.isContentEditable || // Messenger, WhatsApp, etc.
        target.getAttribute("role") === "textbox"
        // Could be more
    ) {
        addCorrectionButton(target);
    }
});

// General function to position the button
function positionButton(inputField, button, buttonDimension) {
    let rect = inputField.getBoundingClientRect();
    button.style.top = `${window.scrollY + rect.top + rect.height / 2 - buttonDimension / 2}px`;
    button.style.left = `${window.scrollX + rect.left + rect.width - buttonDimension}px`;
}

// Manual resize observer for the element (not the whole webpage)
function observeFieldResize(inputField, button) {
    let observer = new ResizeObserver(() => positionButton(inputField, button));
    observer.observe(inputField);
    return observer;
}

function addCorrectionButton(inputField) {
    let existingButton = document.getElementById("correctButton");
    if (existingButton) existingButton.remove();

    let buttonDimension = 24;
    let iconDimension = 16;

    // Create button & styling
    let button = document.createElement("button");
    button.id = "correctButton";

    button.style.position = "absolute";
    button.style.zIndex = "9999";
    button.style.width = `${buttonDimension}px`;
    button.style.height = `${buttonDimension}px`;
    button.style.border = "1px solid darkgreen";
    button.style.background = "white";
    button.style.borderRadius = "50%";
    button.style.display = "flex";
    button.style.alignItems = "center";
    button.style.justifyContent = "center";
    button.style.cursor = "pointer";
    button.style.padding = "0";

    // Image
    let icon = document.createElement("img");
    icon.src = chrome.runtime.getURL("media/icon.png");
    icon.style.width = `${iconDimension}px`;
    icon.style.height = `${iconDimension}px`;
    button.appendChild(icon);

    // Position & event listeners
    positionButton(inputField, button, buttonDimension);
    let resizeObserver = observeFieldResize(inputField, button);
    
    let scrollHandler = () => positionButton(inputField, button, buttonDimension);
    let resizeHandler = () => positionButton(inputField, button, buttonDimension);
    
    window.addEventListener("scroll", scrollHandler);
    window.addEventListener("resize", resizeHandler);

    document.body.appendChild(button);

    // Handle correction
    button.onclick = async function (event) {
        event.stopPropagation();
        let correctedText = await fetchCorrectedText(getFieldValue(inputField));
        console.log("Corrected:", correctedText); // Let this be
        setFieldValue(inputField, correctedText);
    };
    
    inputField.addEventListener("focusout", function () {
        setTimeout(() => {
            if (!document.activeElement || document.activeElement !== button) {
                button.remove();
                window.removeEventListener("scroll", scrollHandler);
                window.removeEventListener("resize", resizeHandler);
                resizeObserver.disconnect();
            }
        }, 100);
    });    
}


// Get and set the text
// **
function getFieldValue(field) {
    if (field.tagName === "INPUT" || field.tagName === "TEXTAREA") {
        return field.value;
    } else {
        return field.innerText;
    }
}

function findTextElement(parent) {
    // Get the element with the text itself
    if (parent.innerText.trim().length > 0) return parent;

    let walker = document.createTreeWalker(parent, NodeFilter.SHOW_ELEMENT, null, false);
    while (walker.nextNode()) {
        let node = walker.currentNode;
        if (node.innerText.trim().length > 0) {
            return node;
        }
    }
    return null;
}

function setFieldValue(field, text) {
    if (field.tagName === "INPUT" || field.tagName === "TEXTAREA") {
        field.value = text;
        field.dispatchEvent(new Event("input", { bubbles: true }));
    } else {
        field.focus();

        let targetElement = findTextElement(field);
        if (!targetElement) return;

        let site = window.location.hostname;

        // Messenger requires a lot more than normal sites
        if (site.includes("facebook.com")) {
            // Delete text with backspace
            let existingText = targetElement.innerText.trim();
            for (let i = 0; i < existingText.length; i++) {
                field.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key: "Backspace" }));
                field.dispatchEvent(new InputEvent("input", { bubbles: true }));
                field.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true, key: "Backspace" }));
            }

            // Paste text via ClipboardEvent
            let clipboardEvent = new ClipboardEvent("paste", {
                bubbles: true,
                cancelable: true,
                clipboardData: new DataTransfer(),
            });

            clipboardEvent.clipboardData.setData("text/plain", text);
            field.dispatchEvent(clipboardEvent);
        } else {
            targetElement.innerText = text;
            targetElement.dispatchEvent(new InputEvent("input", { bubbles: true }));

            let range = document.createRange();
            let selection = window.getSelection();
            range.selectNodeContents(targetElement);
            range.collapse(false); // Move cursor to end
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }
}
// **

async function fetchCorrectedText(text) {
    try {
        let response = await fetch("https://localhost:5000/correct", {
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