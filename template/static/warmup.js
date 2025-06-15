/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;

function readNlines(n) {
  const buffer = new Uint8Array(n * LINESIZE);
  const times = [];

  for (let j = 0; j < 10; j++) {
    const start = performance.now();
    for (let i = 0; i < n; i++) {
      // access the buffer to ensure it's loaded into cache
      const temp = buffer[i * LINESIZE];
    }
    const end = performance.now();
    times.push(end - start);
  }

  // sort the times and return the median
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

self.addEventListener("message", function (e) {
  if (e.data.command === "start") {
    const results = [];

    for (let n = 1; n <= 10000000; n *= 10) {
      const time = readNlines(n);
      if (time === undefined) {
        break;
      }
      results.push({ n: n, time: time });
      // Post intermediate results to show progress
      self.postMessage({ status: "progress", results: results });
    }

    self.postMessage({ status: "complete", results: results });
  }
});
