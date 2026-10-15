
## Streaming thread budget optimization

Split CPU postprocessing into a bounded independent pool; default 9 preparation/joint + 3 postprocess + 1 GPU workers. Native library defaults are set before imports. 63 tests and default service init/exit passed. See `visualization/output/thread-comparison-20261015.json` for all nine real batches. Warm sample batch improved from 42.427 s to 37.439/37.036 s; fallback fluctuations remain.
