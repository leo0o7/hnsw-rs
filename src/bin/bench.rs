#[path = "bench/dataset.rs"]
mod dataset;
#[path = "bench/helpers.rs"]
mod helpers;
#[path = "bench/measurement.rs"]
mod measurement;
#[path = "bench/report.rs"]
mod report;

use measurement::{precompute_pq, run_benchmark};
use serde::Deserialize;
use std::{env, error::Error};

const DEFAULT_CONFIG_PATH: &str = "bench-config.toml";

#[derive(Debug, Deserialize)]
pub(crate) struct BenchFile {
    pub(crate) dataset_path: String,
    pub(crate) dimension: usize,
    pub(crate) top_k: usize,
    pub(crate) warmup_queries: usize,
    pub(crate) query_cycles: Option<usize>,
    pub(crate) query_limit: Option<usize>,
    pub(crate) base_limit: Option<usize>,
    pub(crate) load_index_prefix: Option<String>,
    pub(crate) save_index_prefix: Option<String>,
    pub(crate) seed: Option<u64>,
    pub(crate) base_datasets: Option<Vec<String>>,
    pub(crate) query_datasets: Option<Vec<String>>,
    pub(crate) ground_truth_datasets: Option<Vec<String>>,
    pub(crate) output_json: Option<String>,
    pub(crate) quantized: Option<QuantizedConfig>,
    pub(crate) configs: Vec<BenchConfig>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) struct QuantizedConfig {
    pub(crate) quantizers: usize,
    pub(crate) pq_k: usize,
    #[serde(default)]
    pub(crate) pq_oracle: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) struct BenchConfig {
    pub(crate) m: usize,
    pub(crate) m0: usize,
    pub(crate) ef_construction: usize,
    pub(crate) ef_search: usize,
}

impl BenchConfig {
    pub(crate) fn index_path(self, prefix: &str, dimension: usize) -> String {
        format!(
            "{prefix}-dim{dimension}-m{}-m0{}-efc{}-efs{}.bin",
            self.m, self.m0, self.ef_construction, self.ef_search
        )
    }
}

impl BenchFile {
    pub(crate) fn query_cycles(&self) -> usize {
        self.query_cycles.unwrap_or(1).max(1)
    }

    pub(crate) fn warmup_count(&self, query_count: usize) -> usize {
        self.warmup_queries.min(query_count.saturating_sub(1))
    }
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
    let data = dataset::load_bench_data::<DIM>(config)?;
    let pq_data = quantized.map(|quantized| precompute_pq::<DIM, Q>(&data.base, quantized.pq_k));
    report::print_header(config, &data, quantized, pq_data.as_ref());

    let mut runs = config
        .output_json
        .as_ref()
        .map(|_| Vec::with_capacity(config.configs.len()));

    for params in config.configs.iter().copied() {
        let metrics = run_benchmark::<DIM, Q>(&data, params, config, quantized, pq_data.as_ref())?;
        report::print_metrics(params, data.k, &metrics);
        if let Some(runs) = &mut runs {
            runs.push(report::run_entry(params, &metrics));
        }
    }

    if let Some(output_json) = config.output_json.as_deref() {
        report::write_json_report(
            output_json,
            config,
            &data,
            quantized,
            pq_data.as_ref(),
            runs.expect("JSON output should initialize report runs"),
        )?;
        println!("wrote benchmark JSON: {output_json}");
    }

    Ok(())
}
