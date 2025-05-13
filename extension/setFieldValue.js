// @ts-nocheck
window.setFieldValue = (text, start = null, end = null) => {
    field = window.activeField;
    // console.log(field);

    // Helper to build the new full text if start/end are provided
    function buildFullText(orig) {
        if (typeof start === "number" && typeof end === "number") {
            return orig.slice(0, start)
                + text
                + orig.slice(end);
        }
        // no indices -> replace all
        return text;
    }

    if (field.tagName === "INPUT" || field.tagName === "TEXTAREA") {
        let orig = field.value;
        field.value = buildFullText(orig);
        field.dispatchEvent(new Event("input", { bubbles: true }));
    } else {
        // contentEditable
        field.focus();
        let target = findTextElement(field);
        if (!target) return;

        let fullOrig = target.innerText;
        let newFull = buildFullText(fullOrig);
        let site = window.location.hostname;

        if (site.includes("facebook.com")) {
            // Facebook trick: backspace out everything, then paste newFull
            // last part doesn't need to be reinserted, as it is not deleted
            newFull = fullOrig.slice(0, start) + text + (text.endsWith(" ") ? "" : " ");

            // Delete
            for (let i = 0; i < fullOrig.length; i++) {
                field.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key: "Backspace" }));
                field.dispatchEvent(new InputEvent("input", { bubbles: true }));
                field.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true, key: "Backspace" }));
            }

            let pasteEvt = new ClipboardEvent("paste", {
                bubbles: true,
                cancelable: true,
                clipboardData: new DataTransfer(),
            });
            pasteEvt.clipboardData.setData("text/plain", newFull);
            field.dispatchEvent(pasteEvt);
        } else {
            // Standard contentEditable: just overwrite innerText
            target.innerText = newFull;
            target.dispatchEvent(new InputEvent("input", { bubbles: true }));

            // Move cursor to end
            let range = document.createRange();
            let sel = window.getSelection();
            range.selectNodeContents(target);
            range.collapse(false);
            sel.removeAllRanges();
            sel.addRange(range);
        }
    }
    document.getElementById("wordSuggestAnchor")?.remove();
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