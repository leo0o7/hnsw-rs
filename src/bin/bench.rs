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
    pub(crate) ef_searches: Vec<usize>,
    pub(crate) output_json: Option<String>,
    pub(crate) quantized: Option<QuantizedConfig>,
    pub(crate) configs: Vec<BenchConfig>,
    pub(crate) build_parallel: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) struct QuantizedConfig {
    pub(crate) quantizers: usize,
    pub(crate) pq_k: usize,
    #[serde(default)]
    pub(crate) pq_oracle: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
pub(crate) struct BenchConfig {
    pub(crate) m: usize,
    pub(crate) m0: usize,
    pub(crate) ef_construction: usize,
}

impl BenchConfig {
    pub(crate) fn index_path(self, prefix: &str, dimension: usize) -> String {
        format!(
            "{prefix}-dim{dimension}-m{}-m0{}-efc{}.bin",
            self.m, self.m0, self.ef_construction
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
    match config.dimension {
        128 => run_dimension::<128>(&config),
        784 => run_dimension::<784>(&config),
        dimension => Err(format!(
            "unsupported dimension {}; add a match arm in src/bin/bench.rs",
            dimension
        )
        .into()),
    }
}

fn run_dimension<const DIM: usize>(config: &BenchFile) -> Result<(), Box<dyn Error>> {
    match config.quantized {
        None => run::<DIM, 0>(config),
        Some(quantized) => match quantized.quantizers {
            32 => run::<DIM, 32>(config),
            64 => run::<DIM, 64>(config),
            128 if DIM == 128 => run::<DIM, 128>(config),
            196 if DIM == 784 => run::<DIM, 196>(config),
            other => Err(unsupported_quantizers(DIM, other).into()),
        },
    }
}

fn unsupported_quantizers(dimension: usize, quantizers: usize) -> String {
    let supported = match dimension {
        128 => "32, 64, 128",
        784 => "32, 64, 196",
        _ => "none",
    };
    format!(
        "unsupported quantizers {quantizers} for {dimension}D; supported values are {supported}"
    )
}

fn validate_configs(configs: &[BenchConfig], ef_searches: &[usize]) -> Result<(), &'static str> {
    if configs.is_empty() {
        return Err("configs must contain at least one graph configuration");
    }

    if ef_searches.is_empty() || ef_searches.contains(&0) {
        return Err("ef_searches must contain at least one positive value");
    }
    for (index, ef_search) in ef_searches.iter().enumerate() {
        if ef_searches[..index].contains(ef_search) {
            return Err("ef_searches must not contain duplicates");
        }
    }

    for (index, params) in configs.iter().enumerate() {
        if configs[..index].contains(params) {
            return Err("configs must not contain duplicate graph configurations");
        }
    }

    Ok(())
}

fn run<const DIM: usize, const Q: usize>(config: &BenchFile) -> Result<(), Box<dyn Error>> {
    validate_configs(&config.configs, &config.ef_searches)?;
    let quantized = config.quantized;
    let mut data = dataset::load_bench_data::<DIM>(config)?;
    let pq_data = quantized.map(|quantized| precompute_pq::<DIM, Q>(&data.base, quantized.pq_k));
    report::print_header(config, &data, quantized, pq_data.as_ref());

    let mut runs = config.output_json.as_ref().map(|_| {
        Vec::with_capacity(
            config
                .configs
                .len()
                .saturating_mul(config.ef_searches.len()),
        )
    });

    for params in config.configs.iter().copied() {
        let metrics = run_benchmark::<DIM, Q>(
            &mut data,
            params,
            &config.ef_searches,
            config,
            quantized,
            pq_data.as_ref(),
        )?;
        for (run_index, run) in metrics.into_iter().enumerate() {
            report::print_metrics(params, run.ef_search, data.k, &run.metrics, run_index == 0);
            if let Some(runs) = &mut runs {
                runs.push(report::run_entry(params, run.ef_search, &run.metrics));
            }
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

#[cfg(test)]
mod tests {
    use super::{BenchConfig, validate_configs};

    fn config(m: usize) -> BenchConfig {
        BenchConfig {
            m,
            m0: m * 2,
            ef_construction: 200,
        }
    }

    #[test]
    fn accepts_distinct_graph_and_search_configurations() {
        validate_configs(&[config(8), config(16)], &[16, 64]).expect("valid sweep");
    }

    #[test]
    fn rejects_empty_or_zero_search_effort() {
        assert!(validate_configs(&[], &[32]).is_err());
        assert!(validate_configs(&[config(8)], &[]).is_err());
        assert!(validate_configs(&[config(8)], &[0]).is_err());
        assert!(validate_configs(&[config(8)], &[32, 32]).is_err());
        assert!(validate_configs(&[config(8), config(8)], &[32]).is_err());
    }
}
