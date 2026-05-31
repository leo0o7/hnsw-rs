use std::collections::HashSet;
use std::{fs, path::PathBuf};

use super::*;
use rand::RngExt;

const EPSILON: f32 = 1e-6;
const MIN_RECALL: f32 = 0.95;

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize)]
struct Weighted2D {
    x: f32,
    y: f32,
}

impl Distance<2> for Weighted2D {
    fn distance(&self, a: &[f32; 2], b: &[f32; 2]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        self.x * dx * dx + self.y * dy * dy
    }
}

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

fn brute_force_knn<const D: usize>(
    data: &[[f32; D]],
    query: &[f32; D],
    k: usize,
) -> Vec<(usize, f32)> {
    let mut distances: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i, L2Squared.distance(v, query)))
        .collect();

    distances.sort_by(|a, b| a.1.total_cmp(&b.1));
    distances.truncate(k);
    distances
}

fn calculate_recall(graph_results: Vec<(usize, f32)>, brute_results: Vec<(usize, f32)>) -> f32 {
    let graph_set: HashSet<usize> = graph_results.into_iter().map(|(i, _)| i).collect();
    let hits = brute_results
        .iter()
        .filter(|(i, _)| graph_set.contains(i))
        .count();
    hits as f32 / brute_results.len() as f32
}

fn temp_path(name: &str) -> PathBuf {
    let unique = format!(
        "hnsw-{name}-{}-{}.bin",
        std::process::id(),
        rand::rng().next_u64()
    );
    std::env::temp_dir().join(unique)
}

#[test]
fn test_knn() {
    let mut knn = Hnsw::<2, L2Squared>::new_default(5);
    knn.insert([0.0, 0.0]);
    knn.insert([3.0, 3.0]);
    knn.insert([4.0, 4.0]);

    let from = [1.0, 1.0];
    let closest = knn.search(&from, 2);

    assert_eq!(closest.len(), 2);
    let expected_distances = [2.0_f32, 8.0_f32];
    assert!(approx_eq(closest[0].1, expected_distances[0]));
    assert!(approx_eq(closest[1].1, expected_distances[1]));
}

#[test]
fn test_k_larger_than_number_of_entries() {
    let mut knn = Hnsw::<1, L2Squared>::new_default(5);
    knn.insert([1.0]);
    knn.insert([2.0]);

    let closest = knn.search(&[0.0], 3);
    assert_eq!(closest.len(), 2);
}

#[test]
fn test_duplicate() {
    let mut knn = Hnsw::<1, L2Squared>::new_default(5);
    knn.insert([0.0]);
    knn.insert([2.0]);
    knn.insert([2.0]);
    knn.insert([4.0]);

    let from = [2.0];
    let closest = knn.search(&from, 3);

    assert_eq!(closest.len(), 3);
    assert!(approx_eq(closest[0].1, 0.0));
    assert!(approx_eq(closest[1].1, 0.0));
    assert!(approx_eq(closest[2].1, 4.0));
}

#[test]
fn test_empty_graph() {
    let knn = Hnsw::<2, L2Squared>::new_default(2);
    let closest = knn.search(&[1.0, 1.0], 3);
    dbg!(&closest);
    assert!(closest.is_empty());
}

#[test]
fn custom_distance_controls_search_order() {
    let mut knn =
        Hnsw::<2, Weighted2D>::new_seeded(2, 4, 16, 16, 42, Weighted2D { x: 0.0, y: 1.0 });
    knn.insert([10.0, 0.0]);
    knn.insert([0.0, 2.0]);
    knn.insert([0.0, 3.0]);

    let closest = knn.search(&[0.0, 0.0], 1);

    assert_eq!(closest, vec![(0, 0.0)]);
}

#[test]
fn stateful_distance_survives_save_load() {
    let path = temp_path("weighted-distance");
    let mut original =
        Hnsw::<2, Weighted2D>::new_seeded(2, 4, 16, 16, 42, Weighted2D { x: 0.0, y: 1.0 });
    original.insert([10.0, 0.0]);
    original.insert([0.0, 2.0]);
    original.insert([0.0, 3.0]);

    original.save(&path).unwrap();
    let loaded: Hnsw<2, Weighted2D> = Hnsw::load(&path).unwrap();

    assert_eq!(loaded.search(&[0.0, 0.0], 1), vec![(0, 0.0)]);

    fs::remove_file(path).unwrap();
}

#[test]
#[should_panic(expected = "quantized data length must match HNSW index length")]
fn freeze_with_pq_rejects_mismatched_quantized_data() {
    let mut hnsw = Hnsw::<1>::new_default(2);
    hnsw.insert([1.0]);
    hnsw.insert([2.0]);

    let mut pq = pq::ProductQuantizer::<1, 1>::new(1);
    pq.fit(&[[1.0], [2.0]]);

    hnsw.freeze_with_pq(pq, vec![[0]]);
}

#[test]
fn test_save_load_roundtrip_search_and_insert() {
    let path = temp_path("roundtrip");
    let query = [1.0, 1.0];

    let mut original = Hnsw::<2, L2Squared>::new_seeded(5, 10, 128, 32, 42, L2Squared);
    original.insert([0.0, 0.0]);
    original.insert([3.0, 3.0]);
    original.insert([4.0, 4.0]);

    original.save(&path).unwrap();
    let mut loaded = Hnsw::<2, L2Squared>::load(&path).unwrap();

    assert_eq!(loaded.search(&query, 2), original.search(&query, 2));

    original.insert([2.0, 2.0]);
    loaded.insert([2.0, 2.0]);

    assert_eq!(loaded.search(&query, 3), original.search(&query, 3));

    fs::remove_file(path).unwrap();
}

#[test]
fn test_max_connections() {
    let mut h = Hnsw::<2>::new_default(8);

    for i in 0..100 {
        h.insert([i as f32, 0.0]);
    }

    for node in &h.nodes {
        for (lyr, neighs) in node.layers.iter().enumerate() {
            let max = if lyr == 0 { h.M0 } else { h.M };
            assert!(
                neighs.len() <= max,
                "more than max connections at layer {}: {} > {}",
                lyr,
                neighs.len(),
                max
            );
        }
    }
}

#[test]
fn test_no_duplicate_neighbors() {
    let mut h = Hnsw::<2>::new_default(8);

    for i in 0..100 {
        h.insert([i as f32, 0.0]);
    }

    for node in &h.nodes {
        for neighs in &node.layers {
            let mut seen = std::collections::HashSet::new();
            for n in neighs {
                assert!(seen.insert(n.node_index), "duplicate neighbor detected");
            }
        }
    }
}

#[test]
fn test_avg_recall() {
    const N: usize = 10_000;
    const DIMS: usize = 8;
    const K: usize = 5;
    const M: usize = 16;
    const N_RECALL_QUERIES: usize = 1000;

    let mut rng = rand::rng();
    let mut knn = Hnsw::<DIMS, L2Squared>::new_default(M);

    for _ in 0..N {
        let v: [f32; DIMS] = (0..DIMS)
            .map(|_| rng.random_range(-10.0..10.0))
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();
        knn.insert(v);
    }

    let mut total_recall = 0.0;
    for _ in 0..N_RECALL_QUERIES {
        let query: [f32; DIMS] = (0..DIMS)
            .map(|_| rng.random_range(-10.0..10.0))
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap();
        let graph_results = knn.search(&query, K);
        let brute_results = brute_force_knn(&knn.data, &query, K);
        total_recall += calculate_recall(graph_results, brute_results);
    }
    let avg_recall = total_recall / N_RECALL_QUERIES as f32;

    println!("avg recall: {:.1}%", avg_recall * 100.0);

    assert!(avg_recall >= MIN_RECALL);
}
