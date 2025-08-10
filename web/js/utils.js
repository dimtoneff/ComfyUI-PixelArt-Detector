import { app } from "../../scripts/app.js";

export function deepEqual(obj1, obj2) {
    if (typeof obj1 !== typeof obj2) {
        return false
    }
    if (typeof obj1 !== 'object' || obj1 === null || obj2 === null) {
        return obj1 === obj2
    }
    const keys1 = Object.keys(obj1)
    const keys2 = Object.keys(obj2)
    if (keys1.length !== keys2.length) {
        return false
    }
    for (let key of keys1) {
        if (!deepEqual(obj1[key], obj2[key])) {
            return false
        }
    }
    return true
}

let origProps = {};
export const findWidgetByName = (node, name) => node.widgets.find((w) => w.name === name);

export const doesInputWithNameExist = (node, name) => node.inputs ? node.inputs.some((input) => input.name === name) : false;

export function updateNodeHeight(node) { node.setSize([node.size[0], node.computeSize()[1]]); }

export function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget) {
        console.log(`[toggleWidget] Widget not found, skipping.`);
        return;
    }
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }

    widget.hidden = !show;

    // Set type and size for hiding
    widget.type = show ? origProps[widget.name].origType : "wHidden" + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];
    
    // Mark the node as dirty to ensure a redraw
    app.graph.setDirtyCanvas(true, true);

    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, show, ":" + widget.name));
}
