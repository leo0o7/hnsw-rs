# Benchmark Suite

The benchmark suite contains the configs, structured JSON results, plotting script, and generated figures used in the root README.

## Running the suite

Use the benchmark runner as:

```sh
./benchmarks/run.sh <command>
```

Available commands:

- `build` — build the indexes required by the benchmark
- `measure` — run the measurements and write JSON results
- `plot` — regenerate the figures from the existing JSON results
- `all` — run `build`, `measure`, and `plot` in order

The runner sets `RUSTFLAGS="-C target-cpu=native"` unless `RUSTFLAGS` is already defined.

## Benchmark setup

These results were measured on an Apple M3 Pro with saved indexes loaded from disk:

- CPU: Apple M3 Pro
- Memory: 18 GB
- OS: macOS 15.6.1
- Rust: rustc 1.95.0
- Distance: squared L2
- Metric: recall@10
- Warmup queries: 100
- Distinct measured queries: 900
- Query cycles: 100
- Timed searches: 90,000 (900 queries × 100 cycles)

Download the datasets:

```sh
mkdir -p data

curl -L https://ann-benchmarks.com/sift-128-euclidean.hdf5 \
  -o data/sift-128-euclidean.hdf5

curl -L https://ann-benchmarks.com/mnist-784-euclidean.hdf5 \
  -o data/mnist-784-euclidean.hdf5
```

## Running an individual config

The benchmark binary reads a TOML config, builds or loads the requested indexes, and optionally writes a JSON report.

```sh
RUSTFLAGS="-C target-cpu=native" \
  cargo run --release --bin bench -- benchmarks/configs/path/to/config.toml
```

Existing benchmark configs are organized under:

- `configs/build/`: constructs and saves the indexes used by the benchmark suite.
- `configs/measure/`: loads the saved indexes and measures normal or PQ search performance.

## Config file

The benchmark is configured through a TOML file such as `bench-config.toml`.

The fields describe the dataset and how the benchmark should run:

- `dataset_path`: HDF5 file to read
- `dimension`: vector dimension, currently matched in `src/bin/bench.rs`
- `top_k`: number of expected neighbors
- `warmup_queries`: queries excluded from timing
- `query_limit`: number of queries
- `query_cycles`: number of passes
- `load_index_prefix`: path prefix for loading indexes
- `save_index_prefix`: path prefix for saving indexes
- `output_json`: optional JSON report path

The `[[configs]]` entries are the HNSW parameter sets to run.
Each one produces or loads a separate index using the same filename pattern:

```toml
[[configs]]
m = 16
m0 = 32
ef_construction = 128
```

The top-level `ef_searches` list defines a query-time sweep.
Each value is measured against every graph configuration.

If `load_index_prefix` is set, the benchmark loads matching index files from disk.  
If `save_index_prefix` is set instead, it builds the index from the base dataset and writes it to disk after construction.  
If `output_json` is set, the runner writes one pretty-printed JSON report after all configured runs complete successfully. Console output remains unchanged.

## Detailed results

### Query Performance

#### SIFT-1M

|   M |  M0 | ef_construction | ef_search | load s | memory MiB | recall@10 |     QPS | avg ms | p50 ms | p90 ms | p99 ms | max ms |
| --: | --: | --------------: | --------: | -----: | ---------: | --------: | ------: | -----: | -----: | -----: | -----: | -----: |
|  16 |  32 |             128 |        32 |  0.528 |     872.46 |    0.8897 | 13444.5 |  0.074 |  0.075 |  0.090 |  0.109 |  0.361 |
|  16 |  32 |             128 |        64 |  0.560 |     872.46 |    0.9564 |  7784.5 |  0.128 |  0.132 |  0.153 |  0.178 |  0.585 |
|  32 |  64 |             200 |        64 |  0.638 |     999.03 |    0.9783 |  5743.7 |  0.174 |  0.180 |  0.217 |  0.252 |  0.849 |
|  32 |  64 |             200 |       128 |  0.609 |     999.03 |    0.9948 |  3226.1 |  0.310 |  0.323 |  0.391 |  0.443 |  0.996 |

#### MNIST-60k

|   M |  M0 | ef_construction | ef_search | load s | memory MiB | recall@10 |    QPS | avg ms | p50 ms | p90 ms | p99 ms | max ms |
| --: | --: | --------------: | --------: | -----: | ---------: | --------: | -----: | -----: | -----: | -----: | -----: | -----: |
|  16 |  32 |             128 |        32 |  0.089 |     199.02 |    0.9866 | 8017.7 |  0.125 |  0.126 |  0.156 |  0.185 |  0.497 |
|  16 |  32 |             128 |        64 |  0.089 |     199.02 |    0.9981 | 4849.5 |  0.206 |  0.209 |  0.261 |  0.311 |  0.756 |
|  32 |  64 |             200 |        64 |  0.111 |     202.54 |    0.9986 | 3901.2 |  0.256 |  0.261 |  0.335 |  0.400 |  0.744 |
|  32 |  64 |             200 |       128 |  0.107 |     202.54 |    0.9998 | 2413.7 |  0.414 |  0.421 |  0.552 |  0.657 |  1.425 |

### Product Quantization

These results use frozen PQ indexes over the same saved HNSW graphs.
PQ fit and encode are one-time preprocessing costs; search uses ADC distances over compressed vectors.
Each PQ subquantizer is trained with 256 centroids.

#### SIFT-1M PQ

| quantizers | pq fit s | pq encode s |   M |  M0 | ef_construction | ef_search | load s | memory MiB | recall@10 |     QPS | avg ms | p50 ms | p90 ms | p99 ms | max ms |
| ---------: | -------: | ----------: | --: | --: | --------------: | --------: | -----: | ---------: | --------: | ------: | -----: | -----: | -----: | -----: | -----: |
|         32 |   37.184 |       1.399 |  16 |  32 |             128 |        32 |  0.590 |     414.82 |    0.6821 | 13918.1 |  0.072 |  0.072 |  0.087 |  0.106 |  0.365 |
|         32 |   37.184 |       1.399 |  16 |  32 |             128 |        64 |  0.574 |     414.82 |    0.7096 |  8314.7 |  0.120 |  0.122 |  0.142 |  0.178 |  0.620 |
|         32 |   37.184 |       1.399 |  32 |  64 |             200 |        64 |  0.613 |     541.39 |    0.7157 |  6541.0 |  0.153 |  0.157 |  0.188 |  0.228 |  0.557 |
|         32 |   37.184 |       1.399 |  32 |  64 |             200 |       128 |  0.595 |     541.39 |    0.7182 |  3739.2 |  0.267 |  0.276 |  0.335 |  0.395 |  0.786 |
|         64 |   67.984 |       2.135 |  16 |  32 |             128 |        32 |  0.587 |     445.34 |    0.8319 |  9267.3 |  0.108 |  0.109 |  0.129 |  0.154 |  0.461 |
|         64 |   67.984 |       2.135 |  16 |  32 |             128 |        64 |  0.566 |     445.34 |    0.8738 |  5700.6 |  0.175 |  0.180 |  0.208 |  0.239 |  1.501 |
|         64 |   67.984 |       2.135 |  32 |  64 |             200 |        64 |  0.886 |     571.91 |    0.8873 |  4325.3 |  0.231 |  0.238 |  0.286 |  0.338 |  0.903 |
|         64 |   67.984 |       2.135 |  32 |  64 |             200 |       128 |  0.595 |     571.91 |    0.8952 |  2530.4 |  0.395 |  0.408 |  0.499 |  0.575 |  1.704 |
|        128 |  120.994 |       3.795 |  16 |  32 |             128 |        32 |  0.575 |     506.37 |    0.8897 |  6327.3 |  0.158 |  0.160 |  0.187 |  0.220 |  0.471 |
|        128 |  120.994 |       3.795 |  16 |  32 |             128 |        64 |  0.575 |     506.37 |    0.9563 |  3970.5 |  0.252 |  0.259 |  0.298 |  0.334 |  0.730 |
|        128 |  120.994 |       3.795 |  32 |  64 |             200 |        64 |  0.598 |     632.94 |    0.9782 |  2979.8 |  0.336 |  0.347 |  0.417 |  0.473 |  0.911 |
|        128 |  120.994 |       3.795 |  32 |  64 |             200 |       128 |  0.597 |     632.94 |    0.9947 |  1751.0 |  0.571 |  0.595 |  0.725 |  0.808 |  1.821 |

#### MNIST-60k PQ

| quantizers | pq fit s | pq encode s |   M |  M0 | ef_construction | ef_search | load s | memory MiB | recall@10 |    QPS | avg ms | p50 ms | p90 ms | p99 ms | max ms |
| ---------: | -------: | ----------: | --: | --: | --------------: | --------: | -----: | ---------: | --------: | -----: | -----: | -----: | -----: | -----: | -----: |
|        196 |  195.355 |       0.502 |  16 |  32 |             128 |        32 |  0.090 |      31.56 |    0.9296 | 7664.7 |  0.130 |  0.130 |  0.147 |  0.181 |  0.488 |
|        196 |  195.355 |       0.502 |  16 |  32 |             128 |        64 |  0.091 |      31.56 |    0.9373 | 5802.2 |  0.172 |  0.173 |  0.200 |  0.244 |  0.576 |
|        196 |  195.355 |       0.502 |  32 |  64 |             200 |        64 |  0.090 |      35.08 |    0.9374 | 5129.6 |  0.195 |  0.196 |  0.234 |  0.291 |  0.902 |
|        196 |  195.355 |       0.502 |  32 |  64 |             200 |       128 |  0.089 |      35.08 |    0.9393 | 3579.2 |  0.279 |  0.281 |  0.349 |  0.444 |  1.307 |

### Build Performance

These results measure index construction from the base dataset.

Build-time memory is higher because `memory_usage_bytes()` counts `Vec::capacity()`, not `.len()`.
During incremental construction, `Vec`s over-allocate and leave some spare capacity. After loading from disk, `bincode2` reconstructs with exact capacity, so the unused capacity is gone.

| Dataset   |   M |  M0 | ef_construction | build s | inserts/s | build memory MiB |
| --------- | --: | --: | --------------: | ------: | --------: | ---------------: |
| SIFT-1M   |  16 |  32 |             128 | 206.481 |      4843 |          1042.83 |
| SIFT-1M   |  32 |  64 |             200 | 382.948 |      2611 |          1224.88 |
| SIFT-1M   |  32 |  64 |             200 | 397.282 |      2517 |          1224.88 |
| MNIST-60k |  16 |  32 |             128 |  17.760 |      3378 |           222.27 |
| MNIST-60k |  16 |  32 |             128 |  17.990 |      3335 |           222.27 |
| MNIST-60k |  32 |  64 |             200 |  28.616 |      2097 |           227.42 |
| MNIST-60k |  32 |  64 |             200 |  28.888 |      2077 |           227.42 |
