function app() {
  return {
    /* This is the main app object containing all the application state and methods. */
    // The following properties are used to store the state of the application

    // results of cache latency measurements
    latencyResults: null,
    // local collection of trace data
    traceData: [],
    // Local collection of heapmap images
    heatmaps: [],

    // Current status message
    status: "",
    // Is any worker running?
    isCollecting: false,
    // Is the status message an error?
    statusIsError: false,
    // Show trace data in the UI?
    showingTraces: false,

    // Collect latency data using warmup.js worker
    async collectLatencyData() {
      this.latencyData = [];
      this.isCollecting = true;
      this.worker.postMessage({ command: "start" });
    },

    init() {
      this.worker = new Worker("warmup.js");
      this.worker.onmessage = (e) => {
        if (e.data.status === "progress" || e.data.status === "complete") {
          this.latencyData = e.data.results;
        }
        if (e.data.status === "complete") {
          this.isCollecting = false;
        }
      };
    },

    // Collect trace data using worker.js and send to backend
    async collectTraceData() {
       /* 
        * Implement this function to collect trace data.
        * 1. Create a worker to run the sweep function.
        * 2. Collect the trace data from the worker.
        * 3. Send the trace data to the backend for temporary storage and heatmap generation.
        * 4. Fetch the heatmap from the backend and add it to the local collection.
        * 5. Handle errors and update the status.
        */
    },

    // Download the trace data as JSON (array of arrays format for ML)
    async downloadTraces() {
       /* 
        * Implement this function to download the trace data.
        * 1. Fetch the latest data from the backend API.
        * 2. Create a download file with the trace data in JSON format.
        * 3. Handle errors and update the status.
        */
    },

    // Clear all results from the server
    async clearResults() {
      /* 
       * Implement this function to clear all results from the server.
       * 1. Send a request to the backend API to clear all results.
       * 2. Clear local copies of trace data and heatmaps.
       * 3. Handle errors and update the status.
       */
    },
  };
}
