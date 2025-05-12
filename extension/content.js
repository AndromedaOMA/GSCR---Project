// Prerequisites (selection event handling, first render call to add btn)
window.activeField = null;

document.addEventListener("focusin", (event) => {
    let target = event.target;

    // Normal input/textarea OR a custom contenteditable div
    if (
        target.tagName === "TEXTAREA" ||
        (target.tagName === "INPUT" && target.type === "text") ||
        target.isContentEditable || // Messenger, WhatsApp, etc.
        target.getAttribute("role") === "textbox"
        // Could be more
    ) {
        activeField = target;
        addCorrectionButton(target);
    }
});

["mouseup","keyup"].forEach(evt =>
  document.addEventListener(evt, ev => {
    if (!activeField) return;
    const clickX = ev.pageX, clickY = ev.pageY;
    const selInfo = captureSelection(activeField, ev);
    const mainPopUpExists = document.getElementById("wordSuggestAnchor")
    if (selInfo && !mainPopUpExists) {
      showMainPopUp(selInfo, clickX, clickY);
    }
  })
);


// Function to position the button
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
        let correctedText = await window.fetchCheckedAndCorrectedText(getFieldValue(inputField));
        console.log("Corrected:", correctedText.corrected); // Let this be
        setFieldValue(correctedText.corrected);
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