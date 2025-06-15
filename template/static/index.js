document.addEventListener("alpine:init", () => {
  window.Alpine.store("app", {
    // Shared state
    isCollecting: false,
    status: "",
    statusIsError: false,
    
    // Task 1: Latency Collection
    latencyData: [],
    latencyWorker: null,
    
    // Task 2: Trace Collection
    traces: [],
    traceWorker: null,

    // --- Methods ---
    
    // Task 1 Method
    collectLatencyData() {
      this.isCollecting = true;
      this.status = "Collecting latency data...";
      this.latencyData = [];
      this.latencyWorker.postMessage({ command: "start" });
    },

    // Task 2 Methods
    collectTraceData() {
      this.isCollecting = true;
      this.status = "Collecting trace data (this will take 10 seconds)...";
      this.traceWorker.postMessage({ command: "start" });
    },

    async clearResults() {
      this.status = "Clearing results...";
      try {
        const response = await fetch("/clear_results", { method: "POST" });
        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.error || "Failed to clear results.");
        }
        this.traces = [];
        this.latencyData = [];
        this.status = "Results cleared successfully.";
      } catch (error) {
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      }
    },

    downloadTraces() {
        this.status = "Downloading traces...";
        fetch('/download_traces')
            .then(resp => resp.blob())
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'traces.json';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                this.status = "Traces downloaded.";
            })
            .catch(() => {
                this.status = `Error: Could not download traces.`;
                this.statusIsError = true;
            });
    },

    async fetchResults() {
      try {
        const response = await fetch("/get_results");
        if (!response.ok) return;
        const data = await response.json();
        this.traces = data.traces;
      } catch (error) {
        console.log("Could not fetch initial results.");
      }
    },
    
    // --- Initialization ---
    init() {
      // Init worker for Task 1
      this.latencyWorker = new Worker("warmup.js");
      this.latencyWorker.onmessage = (e) => {
        if (e.data.status === "progress") {
          this.latencyData = e.data.results;
        } else if (e.data.status === "complete") {
          this.latencyData = e.data.results;
          this.isCollecting = false;
          this.status = "Latency collection complete.";
        }
      };

      // Init worker for Task 2
      this.traceWorker = new Worker("worker.js");
      this.traceWorker.onmessage = async (e) => {
        if (e.data.status === "complete") {
          this.status = "Trace collected. Generating heatmap...";
          try {
            const response = await fetch("/collect_trace", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ trace: e.data.trace }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || "Failed to process trace.");
            
            this.traces.push(result);
            this.status = "Heatmap generated successfully.";

          } catch (error) {
            this.status = `Error: ${error.message}`;
            this.statusIsError = true;
          } finally {
            this.isCollecting = false;
          }
        }
      };
      
      this.fetchResults();
    },
  });
});
