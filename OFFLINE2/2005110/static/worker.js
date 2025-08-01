/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;
/* Find the L3 size by running `getconf -a | grep CACHE` */
const LLCSIZE = 8 * 1024 * 1024;
/* Collect traces for 10 seconds; you can vary this */
const TIME = 10000;
/* Collect traces every 10ms; you can vary this */
const P = 10; 

function sweep(P) {
  const K = TIME / P;
  const counts = new Array(K).fill(0);
  const buffer = new Uint8Array(LLCSIZE);
  const num_lines = LLCSIZE / LINESIZE;

  for (let k = 0; k < K; k++) {
    const start = performance.now();
    let sweep_counts = 0;
    while (performance.now() - start < P) {
      for (let i = 0; i < num_lines; i++) {
        const temp = buffer[i * LINESIZE];
      }
      sweep_counts++;
    }
    counts[k] = sweep_counts;
  }
  return counts;
}

self.addEventListener("message", function (e) {
  if (e.data.command === "start") {
    const trace = sweep(P);
    self.postMessage({ status: "complete", trace: trace });
  }
});