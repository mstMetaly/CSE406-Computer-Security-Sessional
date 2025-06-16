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
from database import Database, db

app = Flask(__name__)
db = Database(websites=[]) # Initialize with empty list, will be populated by collect.py

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
    label = data.get("label")
    site_idx = data.get("site_idx")

    if not trace_data or len(trace_data) == 0:
        return jsonify({"error": "No trace data provided"}), 400
    if label is None or site_idx is None:
        return jsonify({"error": "Label or site_idx not provided"}), 400
    
    # Save trace to the database
    db.save_trace(website=label, site_idx=site_idx, trace_data=trace_data)

    trace_id = str(uuid.uuid4()) # We generate an ID for the image, but it's not the DB primary key
    
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

    # The UI will get its data from get_results, but we can return this for one-off confirmation
    return jsonify({
        "id": trace_id,
        "img": img_path_relative,
        "stats": stats
    })

@app.route("/get_results", methods=["GET"])
def get_results():
    # In a full app, you might query the DB here.
    # For this task, showing recent traces is fine, and the collect script manages DB state.
    # We will clear the in-memory traces_storage so the UI only shows new traces since last refresh.
    # The true state is in the DB.
    return jsonify({"traces": []}) # Keep UI simple, true state is in DB managed by collect.py

@app.route("/clear_results", methods=["POST"])
def clear_results():
    # This now clears the database and the images
    try:
        session = db.Session()
        session.execute('DELETE FROM fingerprints;')
        session.execute('UPDATE collection_stats SET traces_collected = 0;')
        session.commit()
    except Exception as e:
        return jsonify({"error": f"DB clear failed: {e}"}), 500
    finally:
        session.close()

    # Also delete the physical trace images
    if os.path.exists(TRACES_DIR_PATH):
        for filename in os.listdir(TRACES_DIR_PATH):
            os.remove(os.path.join(TRACES_DIR_PATH, filename))
        
    return jsonify({"message": "All traces and database records cleared."})

@app.route('/download_traces', methods=['GET'])
def download_traces():
    # This now exports the entire database to JSON
    output_path = "dataset.json"
    db.export_to_json(output_path)
    
    if not os.path.exists(output_path):
        return jsonify({"error": "No traces to download."}), 404
    
    return send_file(
        output_path,
        as_attachment=True,
        download_name='dataset.json',
        mimetype='application/json'
    )

if __name__ == "__main__":
    if not os.path.exists(TRACES_DIR_PATH):
        os.makedirs(TRACES_DIR_PATH)
    app.run(debug=True, port=5001)