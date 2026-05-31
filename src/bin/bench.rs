#[path = "bench/dataset.rs"]
mod dataset;
#[path = "bench/helpers.rs"]
mod helpers;

use dataset::{load_ground_truth, load_vectors, open_dataset, open_optional_dataset};
use hdf5::File;
use helpers::{compute_ground_truth, duration_average, mib, ms, percentile, recall_at_k};
use hnsw::{Hnsw, HnswSearcher, L2Squared};
use pq::ProductQuantizer;
use rayon::prelude::*;
use serde::Deserialize;
use std::{
    env,
    error::Error,
    hint::black_box,
    time::{Duration, Instant},
};

const DEFAULT_CONFIG_PATH: &str = "bench-config.toml";
const DEFAULT_BASE_DATASETS: &[&str] = &["train", "base"];
const DEFAULT_QUERY_DATASETS: &[&str] = &["test", "query", "queries"];
const DEFAULT_GROUND_TRUTH_DATASETS: &[&str] = &["neighbors", "knns", "groundtruth"];

#[derive(Debug, Deserialize)]
struct BenchFile {
    dataset_path: String,
    dimension: usize,
    top_k: usize,
    warmup_queries: usize,
    query_cycles: Option<usize>,
    query_limit: Option<usize>,
    base_limit: Option<usize>,
    load_index_prefix: Option<String>,
    save_index_prefix: Option<String>,
    seed: Option<u64>,
    base_datasets: Option<Vec<String>>,
    query_datasets: Option<Vec<String>>,
    ground_truth_datasets: Option<Vec<String>>,
    quantized: Option<QuantizedConfig>,
    configs: Vec<BenchConfig>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct QuantizedConfig {
    quantizers: usize,
    pq_k: usize,
    #[serde(default)]
    pq_oracle: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct BenchConfig {
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
}

impl BenchConfig {
    fn index_path(self, prefix: &str, dimension: usize) -> String {
        format!(
            "{prefix}-dim{dimension}-m{}-m0{}-efc{}-efs{}.bin",
            self.m, self.m0, self.ef_construction, self.ef_search
        )
    }
}

#[derive(Debug)]
struct Metrics {
    build_time: Option<Duration>,
    insert_qps: Option<f64>,
    load_time: Option<Duration>,
    load_path: Option<String>,
    save_time: Option<Duration>,
    save_path: Option<String>,
    memory_bytes: usize,
    query_count: usize,
    qps: f64,
    recall: f64,
    pq_oracle_recall: Option<f64>,
    avg_latency: Duration,
    p50: Duration,
    p90: Duration,
    p99: Duration,
    max_latency: Duration,
}

struct BenchData<const DIM: usize> {
    base_name: String,
    query_name: String,
    base: Vec<[f32; DIM]>,
    queries: Vec<[f32; DIM]>,
    ground_truth: Vec<Vec<usize>>,
    k: usize,
}

struct IndexTimings {
    build_time: Option<Duration>,
    insert_qps: Option<f64>,
    load_time: Option<Duration>,
    load_path: Option<String>,
    save_time: Option<Duration>,
    save_path: Option<String>,
}

struct QueryMetrics {
    query_count: usize,
    qps: f64,
    recall: f64,
    avg_latency: Duration,
    p50: Duration,
    p90: Duration,
    p99: Duration,
    max_latency: Duration,
}

struct PqBenchData<const DIM: usize, const Q: usize> {
    pq: ProductQuantizer<Q, DIM>,
    quantized_data: Vec<[u8; Q]>,
    fit_time: Duration,
    encode_time: Duration,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config_path = env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_CONFIG_PATH.to_owned());
    let config: BenchFile = toml::from_str(&std::fs::read_to_string(&config_path)?)?;

    // TODO: find some better way of doing this
    match (config.dimension, config.quantized) {
        (128, None) => run::<128, 0>(&config, None),
        (784, None) => run::<784, 0>(&config, None),
        (128, Some(quantized)) => match quantized.quantizers {
            32 => run::<128, 32>(&config, Some(quantized)),
            64 => run::<128, 64>(&config, Some(quantized)),
            128 => run::<128, 128>(&config, Some(quantized)),
            other => Err(unsupported_quantizers(other).into()),
        },
        (784, Some(quantized)) => match quantized.quantizers {
            32 => run::<784, 32>(&config, Some(quantized)),
            64 => run::<784, 64>(&config, Some(quantized)),
            196 => run::<784, 196>(&config, Some(quantized)),
            other => Err(unsupported_quantizers(other).into()),
        },
        other => Err(format!(
            "unsupported dimension {}; add a match arm in src/bin/bench.rs",
            other.0
        )
        .into()),
    }
}

fn unsupported_quantizers(quantizers: usize) -> String {
    format!(
        "unsupported quantizers {quantizers}; supported quantized bench values are 64 for 128D and 196 for 784D"
    )
}

fn run<const DIM: usize, const Q: usize>(
    config: &BenchFile,
    quantized: Option<QuantizedConfig>,
) -> Result<(), Box<dyn Error>> {
    let data = load_bench_data::<DIM>(config)?;
    let pq_data = match quantized {
        Some(quantized) => Some(precompute_pq::<DIM, Q>(&data.base, quantized.pq_k)),
        None => None,
    };
    print_header(config, &data, quantized, pq_data.as_ref());

    for params in config.configs.iter().copied() {
        let metrics = run_benchmark::<DIM, Q>(
            &data.base,
            &data.queries,
            &data.ground_truth,
            params,
            data.k,
            config,
            quantized,
            pq_data.as_ref(),
        )?;
        print_metrics(params, data.k, &metrics);
    }

    Ok(())
}

fn load_bench_data<const DIM: usize>(config: &BenchFile) -> Result<BenchData<DIM>, Box<dyn Error>> {
    let file = File::open(&config.dataset_path)?;
    let base_dataset_names = dataset_names(config.base_datasets.as_deref(), DEFAULT_BASE_DATASETS);
    let query_dataset_names =
        dataset_names(config.query_datasets.as_deref(), DEFAULT_QUERY_DATASETS);
    let ground_truth_dataset_names = dataset_names(
        config.ground_truth_datasets.as_deref(),
        DEFAULT_GROUND_TRUTH_DATASETS,
    );
    let (base_name, base_dataset) = open_dataset(&file, &base_dataset_names)?;
    let (query_name, query_dataset) = open_dataset(&file, &query_dataset_names)?;

    let base = load_vectors::<DIM>(&base_name, &base_dataset, config.base_limit)?;
    let queries = load_vectors::<DIM>(&query_name, &query_dataset, config.query_limit)?;

    if base.is_empty() {
        return Err("base dataset is empty".into());
    }
    if queries.is_empty() {
        return Err("query dataset is empty".into());
    }

    let k = config.top_k.min(base.len());
    let ground_truth = match open_optional_dataset(&file, &ground_truth_dataset_names) {
        Some((name, dataset)) if config.base_limit.is_none() => {
            println!("using ground truth dataset '{name}'");
            load_ground_truth(&name, &dataset, queries.len(), k)?
        }
        Some((name, _)) => {
            println!(
                "ignoring ground truth dataset '{name}' because base_limit is set; computing exact recall for the truncated base set"
            );
            compute_ground_truth(&base, &queries, k)
        }
        None => {
            println!("no ground truth dataset found; computing exact recall with brute force");
            compute_ground_truth(&base, &queries, k)
        }
    };

    Ok(BenchData {
        base_name,
        query_name,
        base,
        queries,
        ground_truth,
        k,
    })
}

fn precompute_pq<const DIM: usize, const Q: usize>(
    base: &[[f32; DIM]],
    pq_k: usize,
) -> PqBenchData<DIM, Q> {
    let mut pq = ProductQuantizer::<Q, DIM>::new(pq_k);

    let fit_start = Instant::now();
    pq.fit(base);
    let fit_time = fit_start.elapsed();

    let encode_start = Instant::now();
    let quantized_data = base.par_iter().map(|vector| pq.encode(vector)).collect();
    let encode_time = encode_start.elapsed();

    PqBenchData {
        pq,
        quantized_data,
        fit_time,
        encode_time,
    }
}

fn print_header<const DIM: usize, const Q: usize>(
    config: &BenchFile,
    data: &BenchData<DIM>,
    quantized: Option<QuantizedConfig>,
    pq_data: Option<&PqBenchData<DIM, Q>>,
) {
    println!("dataset: {}", config.dataset_path);
    println!("base: {} ({} vectors)", data.base_name, data.base.len());
    println!(
        "queries: {} ({} vectors)",
        data.query_name,
        data.queries.len()
    );
    println!("dimension: {DIM}");
    if let Some(quantized) = quantized {
        println!("quantizers: {}", quantized.quantizers);
        println!("pq k: {}", quantized.pq_k);
        if let Some(pq_data) = pq_data {
            println!("pq fit: {:.3}s", pq_data.fit_time.as_secs_f64());
            println!("pq encode: {:.3}s", pq_data.encode_time.as_secs_f64());
        }
    }
    println!("recall metric: recall@{}", data.k);
    println!(
        "warmup queries: {}",
        config
            .warmup_queries
            .min(data.queries.len().saturating_sub(1))
    );
    println!("query cycles: {}", config.query_cycles());
    println!();
}

fn dataset_names<'a>(configured: Option<&'a [String]>, defaults: &[&'a str]) -> Vec<&'a str> {
    configured
        .map(|names| names.iter().map(String::as_str).collect())
        .unwrap_or_else(|| defaults.to_vec())
}

#[allow(clippy::too_many_arguments)]
fn run_benchmark<const DIM: usize, const Q: usize>(
    base: &[[f32; DIM]],
    queries: &[[f32; DIM]],
    ground_truth: &[Vec<usize>],
    params: BenchConfig,
    k: usize,
    config: &BenchFile,
    quantized: Option<QuantizedConfig>,
    pq_data: Option<&PqBenchData<DIM, Q>>,
) -> Result<Metrics, Box<dyn Error>> {
    let (index, mut timings) = load_or_build_index(base, params, config)?;
    save_index(&index, params, DIM, config, &mut timings)?;

    if let Some(quantized) = quantized {
        let pq_data = pq_data.ok_or("quantized benchmark is missing precomputed PQ data")?;
        let index = index.freeze_with_pq(pq_data.pq.clone(), pq_data.quantized_data.clone());
        let warmup = config.warmup_queries.min(queries.len().saturating_sub(1));
        let pq_oracle_recall = if quantized.pq_oracle {
            let measured_queries = queries.len() - warmup;
            let recall_sum: f64 = queries
                .iter()
                .zip(ground_truth)
                .skip(warmup)
                .map(|(query, expected)| recall_at_k(expected, &index.brute_force_adc(query, k)))
                .sum();
            Some(recall_sum / measured_queries as f64)
        } else {
            None
        };

        Ok(measure_index(
            &index,
            timings,
            queries,
            ground_truth,
            k,
            config,
            pq_oracle_recall,
        ))
    } else {
        Ok(measure_index(
            &index,
            timings,
            queries,
            ground_truth,
            k,
            config,
            None,
        ))
    }
}

fn load_or_build_index<const DIM: usize>(
    base: &[[f32; DIM]],
    params: BenchConfig,
    config: &BenchFile,
) -> Result<(Hnsw<DIM>, IndexTimings), Box<dyn Error>> {
    let load_path = config
        .load_index_prefix
        .as_deref()
        .map(|prefix| params.index_path(prefix, DIM));
    let (index, build_time, insert_qps, load_time) = match &load_path {
        Some(path) => {
            let load_start = Instant::now();
            let index = Hnsw::<DIM>::load(path)?;
            let load_time = load_start.elapsed();
            if index.len() != base.len() {
                return Err(format!(
                    "loaded index '{path}' has {} vectors, but benchmark base has {} vectors; use a matching index or remove load_index_prefix",
                    index.len(),
                    base.len()
                )
                .into());
            }
            (index, None, None, Some(load_time))
        }
        None => {
            let mut index = Hnsw::<DIM>::new_seeded(
                params.m,
                params.m0,
                params.ef_construction,
                params.ef_search,
                config.seed.unwrap_or(42),
                L2Squared,
            );

            let build_start = Instant::now();
            let mut insert_ctx = index.insert_context();
            for &vector in base {
                index.insert_with_context(vector, &mut insert_ctx);
            }
            let build_time = build_start.elapsed();
            let insert_qps = base.len() as f64 / build_time.as_secs_f64();
            (index, Some(build_time), Some(insert_qps), None)
        }
    };

    let timings = IndexTimings {
        build_time,
        insert_qps,
        load_time,
        load_path,
        save_time: None,
        save_path: None,
    };
    Ok((index, timings))
}

fn save_index<const DIM: usize>(
    index: &Hnsw<DIM>,
    params: BenchConfig,
    dimension: usize,
    config: &BenchFile,
    timings: &mut IndexTimings,
) -> Result<(), Box<dyn Error>> {
    timings.save_path = config
        .save_index_prefix
        .as_deref()
        .map(|prefix| params.index_path(prefix, dimension));
    if let Some(path) = &timings.save_path {
        let save_start = Instant::now();
        index.save(path)?;
        timings.save_time = Some(save_start.elapsed());
    }

    Ok(())
}

fn measure_index<const DIM: usize, S: HnswSearcher<DIM>>(
    index: &S,
    timings: IndexTimings,
    queries: &[[f32; DIM]],
    ground_truth: &[Vec<usize>],
    k: usize,
    config: &BenchFile,
    pq_oracle_recall: Option<f64>,
) -> Metrics {
    let memory_bytes = index.memory_usage_bytes();
    let warmup = config.warmup_queries.min(queries.len().saturating_sub(1));
    let mut search_ctx = index.search_context();
    let query_metrics = measure_queries(
        queries,
        ground_truth,
        warmup,
        config.query_cycles(),
        |query| index.search_with_context(query, k, &mut search_ctx),
    );

    Metrics {
        build_time: timings.build_time,
        insert_qps: timings.insert_qps,
        load_time: timings.load_time,
        load_path: timings.load_path,
        save_time: timings.save_time,
        save_path: timings.save_path,
        memory_bytes,
        query_count: query_metrics.query_count,
        qps: query_metrics.qps,
        recall: query_metrics.recall,
        pq_oracle_recall,
        avg_latency: query_metrics.avg_latency,
        p50: query_metrics.p50,
        p90: query_metrics.p90,
        p99: query_metrics.p99,
        max_latency: query_metrics.max_latency,
    }
}

fn measure_queries<const DIM: usize>(
    queries: &[[f32; DIM]],
    ground_truth: &[Vec<usize>],
    warmup: usize,
    query_cycles: usize,
    mut search: impl FnMut(&[f32; DIM]) -> Vec<(usize, f32)>,
) -> QueryMetrics {
    for query in queries.iter().take(warmup) {
        let _ = black_box(search(black_box(query)));
    }

    let measured_queries = queries.len() - warmup;
    let total_measured_queries = measured_queries * query_cycles;
    let mut total_search_time = Duration::ZERO;
    let mut recall_sum = 0.0;
    let mut latencies = Vec::with_capacity(total_measured_queries);

    for _ in 0..query_cycles {
        for (query, expected) in queries.iter().zip(ground_truth).skip(warmup) {
            let start = Instant::now();
            let result = search(black_box(query));
            let latency = start.elapsed();

            total_search_time += latency;
            recall_sum += recall_at_k(expected, &result);
            latencies.push(latency);
            black_box(result);
        }
    }

    latencies.sort_unstable();
    QueryMetrics {
        query_count: total_measured_queries,
        qps: total_measured_queries as f64 / total_search_time.as_secs_f64(),
        recall: recall_sum / total_measured_queries as f64,
        avg_latency: duration_average(total_search_time, total_measured_queries),
        p50: percentile(&latencies, 0.50),
        p90: percentile(&latencies, 0.90),
        p99: percentile(&latencies, 0.99),
        max_latency: *latencies.last().unwrap(),
    }
}

impl BenchFile {
    fn query_cycles(&self) -> usize {
        self.query_cycles.unwrap_or(1).max(1)
    }
}

fn print_metrics(params: BenchConfig, k: usize, metrics: &Metrics) {
    println!(
        "M={} M0={} ef_construction={} ef_search={}",
        params.m, params.m0, params.ef_construction, params.ef_search
    );
    if let Some(load_time) = metrics.load_time {
        if let Some(path) = &metrics.load_path {
            println!("  load: {:.3}s ({path})", load_time.as_secs_f64());
        } else {
            println!("  load: {:.3}s", load_time.as_secs_f64());
        }
    }
    if let (Some(build_time), Some(insert_qps)) = (metrics.build_time, metrics.insert_qps) {
        println!(
            "  build: {:.3}s ({:.0} inserts/s)",
            build_time.as_secs_f64(),
            insert_qps,
        );
    }
    if let Some(save_time) = metrics.save_time {
        if let Some(path) = &metrics.save_path {
            println!("  save: {:.3}s ({path})", save_time.as_secs_f64());
        } else {
            println!("  save: {:.3}s", save_time.as_secs_f64());
        }
    }
    println!(
        "  memory: {:.2} MiB ({} bytes)",
        mib(metrics.memory_bytes),
        metrics.memory_bytes,
    );
    println!(
        "  search: recall@{k} {:.4}, {:.1} QPS over {} measured queries",
        metrics.recall, metrics.qps, metrics.query_count,
    );
    if let Some(pq_oracle_recall) = metrics.pq_oracle_recall {
        println!(
            "  diagnostic: pq_bruteforce_oracle_recall@{k} {:.4}, frozen_hnsw_adc_recall@{k} {:.4}",
            pq_oracle_recall, metrics.recall,
        );
    }
    println!(
        "  latency: avg {:.3} ms, p50 {:.3} ms, p90 {:.3} ms, p99 {:.3} ms, max {:.3} ms",
        ms(metrics.avg_latency),
        ms(metrics.p50),
        ms(metrics.p90),
        ms(metrics.p99),
        ms(metrics.max_latency),
    );
    println!();
}
