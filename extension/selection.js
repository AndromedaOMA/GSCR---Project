let lastSelection = { start: null, end: null, text: "" };

window.captureSelection = (field, ev) => {
    let raw = "", start = null, end = null;

    if (field.tagName === "INPUT" || field.tagName === "TEXTAREA") {
        const s = field.selectionStart, e = field.selectionEnd;
        if (s === e) return null;
        raw = field.value.slice(s, e);
        start = s; end = e;
    } else {
        const sel = window.getSelection();
        if (!sel.rangeCount) return null;
        const range = sel.getRangeAt(0);
        raw = sel.toString();
        if (!raw) return null;
        start = range.startOffset;
        end = range.endOffset;
    }

    // trim whitespace
    const leadingWS = raw.match(/^\s*/)[0].length;
    const trailingWS = raw.match(/\s*$/)[0].length;
    const text = raw.trim();
    if (!text) return null;

    start += leadingWS;
    end -= trailingWS;

    const newSel = { start, end, text };
    lastSelection = newSel;
    return newSel;
};