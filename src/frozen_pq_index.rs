use pq::ProductQuantizer;
use rayon::prelude::*;

use crate::{
    Hnsw, HnswSearcher, L2Squared, context::SearchContext, link::Link, node::Node,
    nodes_heap_usage_bytes,
};
use std::{cell::Cell, cmp::Reverse, mem::size_of};

#[allow(non_snake_case)]
pub struct FrozenPQHnsw<const D: usize, const Q: usize> {
    ef_search: usize,
    entry_point: usize,
    data: Vec<[u8; Q]>,
    nodes: Vec<Node>,
    max_layer: usize,
    epoch: Cell<usize>,
    pq: ProductQuantizer<Q, D>,
}

impl<const D: usize, const Q: usize> FrozenPQHnsw<D, Q> {
    pub(crate) fn from_pq(
        hnsw: Hnsw<D, L2Squared>,
        quantized_data: Vec<[u8; Q]>,
        pq: ProductQuantizer<Q, D>,
    ) -> Self {
        assert_eq!(
            quantized_data.len(),
            hnsw.len(),
            "quantized data length must match HNSW index length"
        );
        Self {
            ef_search: hnsw.ef_search,
            entry_point: hnsw.entry_point,
            data: quantized_data,
            nodes: hnsw.nodes,
            max_layer: hnsw.max_layer,
            epoch: hnsw.epoch,
            pq,
        }
    }

    pub(crate) fn from_hnsw(mut hnsw: Hnsw<D, L2Squared>, k: usize) -> Self {
        let mut pq: ProductQuantizer<Q, D> = ProductQuantizer::new(k);
        let hnsw_data = std::mem::take(&mut hnsw.data);
        pq.fit(&hnsw_data);

        Self::from_pq(
            hnsw,
            hnsw_data.into_par_iter().map(|v| pq.encode(&v)).collect(),
            pq,
        )
    }

    fn search_layer_with_context<'a>(
        &self,
        adc_table: &[f32],
        ep: usize,
        lyr: usize,
        ef: usize,
        ctx: &'a mut SearchContext,
    ) -> &'a [Link] {
        ctx.clear();
        let frontier = &mut ctx.frontier;
        let best = &mut ctx.best;
        self.epoch.set(self.epoch.get() + 1);

        let ep_link = Link {
            node_index: ep,
            distance: self.pq.adc_distance(adc_table, &self.data[ep]),
        };
        frontier.push(Reverse(ep_link));
        best.push(ep_link);
        self.nodes[ep].epoch.set(self.epoch.get());

        while let Some(Reverse(candidate)) = frontier.pop() {
            let furthest_dist = best.peek().map_or(f32::INFINITY, |l| l.distance);
            if candidate.distance > furthest_dist {
                break;
            }
            for neigh in self.nodes[candidate.node_index].layers[lyr].iter() {
                if self.nodes[neigh.node_index].epoch == self.epoch {
                    continue;
                }
                self.nodes[neigh.node_index].epoch.set(self.epoch.get());
                let dist = self
                    .pq
                    .adc_distance(adc_table, &self.data[neigh.node_index]);
                if best.len() == ef && best.peek().is_some_and(|furthest| furthest.distance > dist)
                {
                    best.pop();
                }
                if best.len() < ef {
                    let link = Link {
                        node_index: neigh.node_index,
                        distance: dist,
                    };
                    best.push(link);
                    frontier.push(Reverse(link));
                }
            }
        }

        self.avoid_epoch_overflow();
        ctx.consume_best()
    }

    #[inline(always)]
    /// very unlikely to ever be required
    /// this can happen only after 2^64 - 1 epochs
    fn avoid_epoch_overflow(&self) {
        if self.epoch.get() == usize::MAX {
            self.epoch.set(0);
            for node in self.nodes.iter() {
                node.epoch.set(0);
            }
        }
    }

    pub fn brute_force_adc(&self, q: &[f32; D], k: usize) -> Vec<(usize, f32)> {
        let adc = self.pq.adc_table(q);
        let mut distances: Vec<(usize, f32)> = self
            .data
            .iter()
            .enumerate()
            .map(|(id, code)| (id, self.pq.adc_distance(&adc, code)))
            .collect();
        distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        distances.truncate(k);
        distances
    }
}

impl<const D: usize> Hnsw<D, L2Squared> {
    pub fn freeze<const Q: usize>(self, k: usize) -> FrozenPQHnsw<D, Q> {
        FrozenPQHnsw::from_hnsw(self, k)
    }
    pub fn freeze_with_pq<const Q: usize>(
        self,
        pq: ProductQuantizer<Q, D>,
        quantized_data: Vec<[u8; Q]>,
    ) -> FrozenPQHnsw<D, Q> {
        FrozenPQHnsw::from_pq(self, quantized_data, pq)
    }
}

impl<const D: usize, const Q: usize> HnswSearcher<D> for FrozenPQHnsw<D, Q> {
    fn search_context(&self) -> SearchContext {
        SearchContext::init(self.ef_search)
    }

    fn search(&self, q: &[f32; D], k: usize) -> Vec<(usize, f32)> {
        let mut ctx = self.search_context();
        self.search_with_context(q, k, &mut ctx)
    }

    fn search_with_context(
        &self,
        q: &[f32; D],
        k: usize,
        ctx: &mut SearchContext,
    ) -> Vec<(usize, f32)> {
        if self.data.is_empty() {
            return Vec::new();
        }

        let adc = self.pq.adc_table(q);
        let mut ep = self.entry_point;
        for lyr in (1..=self.max_layer).rev() {
            ep = self
                .search_layer_with_context(&adc, ep, lyr, 1, ctx)
                .first()
                .unwrap_or_else(|| {
                    panic!("ERROR: search_layer@{lyr} returned an empty array (search)")
                })
                .node_index;
        }

        let results = self.search_layer_with_context(&adc, ep, 0, self.ef_search.max(k), ctx);
        // take k best from final layer search
        results[..k.min(results.len())]
            .iter()
            .map(|l| (l.node_index, l.distance))
            .collect()
    }

    fn memory_usage_bytes(&self) -> usize {
        size_of::<Self>()
            + self.data.capacity() * size_of::<[u8; Q]>()
            + nodes_heap_usage_bytes(&self.nodes)
            + self.pq.heap_usage_bytes()
    }
}
