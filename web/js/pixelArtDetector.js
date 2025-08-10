import { app } from "../../scripts/app.js";
import { findWidgetByName, doesInputWithNameExist, updateNodeHeight, toggleWidget } from './utils.js'

function togglePixelizeWidgets(node, pixelizeValue) {
    const widgetsConfig = {
        "Image.quantize": ["image_quantize_reduce_method"],
        "Grid.pixelate": ["grid_pixelate_grid_scan_size"],
        "NP.quantize": [],
        "OpenCV.kmeans.reduce": [
            "opencv_settings",
            "opencv_kmeans_centers",
            "opencv_kmeans_attempts",
            "opencv_criteria_max_iterations",
        ],
        "Pycluster.kmeans.reduce": ["pycluster_kmeans_metrics"],
        "Pycluster.kmedians.reduce": ["pycluster_kmeans_metrics"],
    };

    const allToggleableWidgets = [...new Set(Object.values(widgetsConfig).flat())];

    allToggleableWidgets.forEach(widgetName => {
        const widget = findWidgetByName(node, widgetName);
        if (widget) {
            toggleWidget(node, widget, false);
        }
    });

    const widgetsToShow = widgetsConfig[pixelizeValue];
    if (widgetsToShow) {
        widgetsToShow.forEach(widgetName => {
            const widget = findWidgetByName(node, widgetName);
            if (widget) {
                toggleWidget(node, widget, true);
            } else {
                console.log("Widget not found:", widgetName);
            }
        });
    }

    updateNodeHeight(node);
}


app.registerExtension({
	name: "dimtoneff.pixelArtDetector",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PixelArtDetectorConverter") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const onNodeCreatedResult = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const pixelizeWidget = this.widgets.find((w) => w.name === "pixelize");
                if(pixelizeWidget) {
                    // Store original callback
                    const originalCallback = pixelizeWidget.callback;
                    pixelizeWidget.callback = (value) => {
                        // Call original callback if it exists
                        if(originalCallback) {
                            originalCallback.call(this, value);
                        }
                        // Toggle widgets
                        togglePixelizeWidgets(this, value);
                    };
                }

                // Set initial visibility
                setTimeout(() => {
                    if (pixelizeWidget) {
                        togglePixelizeWidgets(this, pixelizeWidget.value);
                    }
                }, 1);

                return onNodeCreatedResult;
            };
        }
    },
	setup(app,file){
async function getWebpExifData(webpFile) {
	const reader = new FileReader();
	reader.readAsArrayBuffer(webpFile);
  
	return new Promise((resolve, reject) => {
	  reader.onloadend = function() {
		const buffer = reader.result;
		const view = new DataView(buffer);
		let offset = 0;
  
		// Search for the "EXIF" tag
		while (offset < view.byteLength - 4) {
		  if (view.getUint32(offset, true) === 0x46495845 /* "EXIF" in big-endian */) {
			const exifOffset = offset + 6; 
			const exifData = buffer.slice(exifOffset);
			const exifString = new TextDecoder().decode(exifData).replaceAll(String.fromCharCode(0), ''); //Remove Null Terminators from string
			let exifJsonString = exifString.slice(exifString.indexOf("Workflow")); //find beginning of Workflow Exif Tag
			let promptregex="(?<!\{)}Prompt:{(?![\w\s]*[\}])"; //Regex to split }Prompt:{ // Hacky as fuck - theoretically if somebody has a text encode with dynamic prompts turned off, they could enter }Prompt:{ which breaks this
			let exifJsonStringMap = new Map([
			
			["workflow",exifJsonString.slice(9,exifJsonString.search(promptregex)+1)], // Remove "Workflow:" keyword in front of the JSON workflow data passed
			["prompt",exifJsonString.substring((exifJsonString.search(promptregex)+8))] //Find and remove "Prompt:" keyword in front of the JSON prompt data

			]);
			let fullJson=Object.fromEntries(exifJsonStringMap); //object to pass back
			
			resolve(fullJson);
			
		  }
  
		  offset++;
		}
  
		reject(new Error('EXIF metadata not found'));
}})};

async function getJpegExifData(file) {
	const reader = new FileReader();
	reader.readAsArrayBuffer(file);
  
	return new Promise((resolve, reject) => {
	  reader.onloadend = function() {
		const BUFFER = reader.result;
		const VIEW = new DataView(BUFFER);
		const PROMPT_REGEX = "(?<!\{)}Prompt:{(?![\w\s]*[\}])";
		const START_EXIF = 1802661719;
		const END_EXIF = 8224034;
		let offset = 0;
		let startOffset = -1;
		let endOffset = -1;
  
		// Search for the "EXIF" tag
		while (offset < VIEW.byteLength - 4) {
			if (VIEW.getUint32(offset, true) === START_EXIF) {
				startOffset = offset;
			}

			if (VIEW.getUint32(offset, true) === END_EXIF) {
				endOffset = offset;
				break;
			}

		  offset++;
		}

		if (startOffset == -1 || endOffset == -1) {  
			reject(new Error('EXIF metadata not found'));
		}

		const exifData = BUFFER.slice(startOffset, endOffset);
		const exifJsonString = new TextDecoder().decode(exifData).replaceAll(String.fromCharCode(0), ''); //Remove Null Terminators from string

		let exifJsonStringMap = new Map([
			
			["workflow",exifJsonString.slice(9,exifJsonString.search(PROMPT_REGEX)+1)], // Remove "Workflow:" keyword in front of the JSON workflow data passed
			["prompt",exifJsonString.substring((exifJsonString.search(PROMPT_REGEX)+8))] //Find and remove "Prompt:" keyword in front of the JSON prompt data

		]);
		let fullJson=Object.fromEntries(exifJsonStringMap); //object to pass back

		resolve(fullJson);
}})};

const handleFile = app.handleFile;
app.handleFile = async function(file) { // Add the 'file' parameter to the function definition
	if (file.type === "image/webp") {
		
		const webpInfo =await getWebpExifData(file);
		if (webpInfo) {
			if (webpInfo.workflow) {
				if(app.load_workflow_with_components) {
					app.load_workflow_with_components(webpInfo.workflow);
				}
				else
					this.loadGraphData(JSON.parse(webpInfo.workflow));
			}
		}
	} else if (file.type === "image/jpeg") {
		const jpegInfo =await getJpegExifData(file);
		if (jpegInfo) {
			if (jpegInfo.workflow) {
				if(app.load_workflow_with_components) {
					app.load_workflow_with_components(jpegInfo.workflow);
				}
				else
					this.loadGraphData(JSON.parse(jpegInfo.workflow));
			}
		}
	} else {
		return handleFile.apply(this, arguments);
	}
}},});
