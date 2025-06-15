import os
import shutil
import uuid
import json
from flask import Flask, jsonify, request, send_file, send_from_directory
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# --- Configuration ---
DATABASE = 'database.db'
# Note: This directory is relative to the 'static' folder
TRACES_DIR_NAME = 'traces'
TRACES_DIR_PATH = os.path.join('static', TRACES_DIR_NAME)

# This will store our data in memory
# For a real application, you'd use a database
traces_storage = {}

@app.route("/")
def index():
    return send_from_directory('static', 'index.html')

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory('static', path)

@app.route("/collect_trace", methods=["POST"])
def collect_trace():
    data = request.get_json()
    trace_data = data.get("trace")
    if not trace_data or len(trace_data) == 0:
        return jsonify({"error": "No trace data provided"}), 400

    trace_id = str(uuid.uuid4())
    
    # --- Generate and save heatmap ---
    try:
        if not os.path.exists(TRACES_DIR_PATH):
            os.makedirs(TRACES_DIR_PATH)

        # Reshape for 1D heatmap
        trace_np = np.array(trace_data)
        heatmap_data = trace_np.reshape(1, -1)
        
        # Create heatmap plot
        fig, ax = plt.subplots(figsize=(10, 1)) # Make it long and thin
        ax.imshow(heatmap_data, cmap='viridis', interpolation='nearest', aspect='auto')
        plt.axis('off')
        fig.tight_layout(pad=0)

        # Save image to a static path
        img_path_relative = os.path.join(TRACES_DIR_NAME, f"{trace_id}.png")
        img_path_full = os.path.join('static', img_path_relative)
        plt.savefig(img_path_full, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Calculate stats
        stats = {
            "min": int(trace_np.min()),
            "max": int(trace_np.max()),
            "range": int(trace_np.max() - trace_np.min()),
            "samples": len(trace_np)
        }

    except Exception as e:
        return jsonify({"error": f"Failed to generate heatmap: {e}"}), 500

    # Store data
    traces_storage[trace_id] = {
        "id": trace_id,
        "raw_trace": trace_data,
        "img": img_path_relative,
        "stats": stats
    }

    return jsonify(traces_storage[trace_id])

@app.route("/get_results", methods=["GET"])
def get_results():
    return jsonify({"traces": list(traces_storage.values())})

@app.route("/clear_results", methods=["POST"])
def clear_results():
    global traces_storage
    traces_storage = {}
    
    # Also delete the physical trace images
    if os.path.exists(TRACES_DIR_PATH):
        shutil.rmtree(TRACES_DIR_PATH)
        
    return jsonify({"message": "All traces cleared."})

@app.route('/download_traces', methods=['GET'])
def download_traces():
    # We'll just download the raw trace data
    all_raw_traces = [data['raw_trace'] for data in traces_storage.values()]
    
    if not all_raw_traces:
        return jsonify({"error": "No traces to download."}), 404
        
    # Use io.BytesIO to create an in-memory binary file
    mem_file = io.BytesIO()
    mem_file.write(json.dumps(all_raw_traces, indent=2).encode('utf-8'))
    mem_file.seek(0)
    
    return send_file(
        mem_file,
        as_attachment=True,
        download_name='traces.json',
        mimetype='application/json'
    )

if __name__ == "__main__":
    if not os.path.exists(TRACES_DIR_PATH):
        os.makedirs(TRACES_DIR_PATH)
    app.run(debug=True, port=5001)