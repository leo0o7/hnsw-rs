use rand::{distr::Open01, prelude::*};

use crate::{
    context::{InsertContext, SearchContext, SelectContext},
    link::Link,
    node::{Node, nodes_heap_usage_bytes},
};
use std::{cell::Cell, cmp::Reverse, mem::size_of};

mod context;
mod disk;
mod dist;
mod frozen_pq_index;
mod link;
mod node;
#[cfg(test)]
mod tests;

pub use dist::{Distance, L2Squared};

#[allow(non_snake_case)]
pub struct Hnsw<const D: usize, DS = L2Squared> {
    M: usize,
    M0: usize,
    pub(crate) ef_construction: usize,
    ef_search: usize,
    pub(crate) entry_point: usize,
    pub(crate) data: Vec<[f32; D]>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) max_layer: usize,
    epoch: Cell<usize>,
    ml: f64,
    seed: u64,
    rng: StdRng,
    dist: DS,
}

pub trait HnswSearcher<const D: usize> {
    fn search_context(&self) -> SearchContext;

    fn search(&self, q: &[f32; D], k: usize) -> Vec<(usize, f32)>;

    fn search_with_context(
        &self,
        q: &[f32; D],
        k: usize,
        ctx: &mut SearchContext,
    ) -> Vec<(usize, f32)>;

    fn memory_usage_bytes(&self) -> usize;
}

impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D>,
{
    #[allow(non_snake_case)]
    pub fn new_seeded(
        M: usize,
        M0: usize,
        ef_construction: usize,
        ef_search: usize,
        seed: u64,
        dist: DS,
    ) -> Self {
        assert!(M > 1, "M must be > 1");
        assert!(M0 > 0, "M0 must be > 0");
        assert!(ef_construction > 0, "ef_construction must be > 0");
        assert!(ef_search > 0, "ef_search must be > 0");

        let ml = 1.0 / (M as f64).ln();
        Self {
            M,
            M0,
            ef_construction,
            ef_search,
            entry_point: 0,
            data: Vec::new(),
            nodes: Vec::new(),
            max_layer: 0,
            epoch: Cell::new(0),
            ml,
            seed,
            rng: StdRng::seed_from_u64(seed),
            dist,
        }
    }

    #[allow(non_snake_case)]
    pub fn new(M: usize, M0: usize, ef_construction: usize, ef_search: usize, dist: DS) -> Self {
        let seed = rand::rng().next_u64();
        Self::new_seeded(M, M0, ef_construction, ef_search, seed, dist)
    }

    pub fn insert_context(&self) -> InsertContext {
        InsertContext {
            select_ctx: SelectContext::init(self.M0),
            search_ctx: SearchContext::init(self.ef_construction),
        }
    }

    pub fn insert(&mut self, vec: [f32; D]) {
        let mut ctx = self.insert_context();
        self.insert_with_context(vec, &mut ctx);
    }

    pub fn insert_with_context(&mut self, vec: [f32; D], ctx: &mut InsertContext) {
        let insert_idx = self.data.len();
        let insert_lyr = self.random_layer();
        self.data.push(vec);
        self.nodes.push(Node {
            layers: Vec::with_capacity(insert_lyr + 1),
            epoch: Cell::new(0),
        });
        for lyr in 0..=insert_lyr {
            let max_connections = self.max_connections(lyr);
            self.nodes[insert_idx]
                .layers
                .push(Vec::with_capacity(max_connections));
        }

        if insert_idx == 0 {
            self.entry_point = 0;
            self.max_layer = insert_lyr;
            return;
        }

        let insert_ctx = &mut ctx.search_ctx;
        let mut ep = self.entry_point;
        for lyr in ((insert_lyr + 1)..=self.max_layer).rev() {
            ep = self
                .search_layer_with_context(&vec, ep, lyr, 1, insert_ctx)
                .first()
                .unwrap_or_else(|| {
                    panic!("ERROR: search_layer@{lyr} returned an empty array (insert)")
                })
                .node_index;
        }

        let select_ctx = &mut ctx.select_ctx;
        for lyr in (0..=insert_lyr.min(self.max_layer)).rev() {
            let candidates =
                self.search_layer_with_context(&vec, ep, lyr, self.ef_construction, insert_ctx);
            let neighs = self.select_neighbors(&vec, lyr, candidates, false, false, select_ctx);
            self.nodes[insert_idx].layers[lyr] = neighs;

            // can't use .iter() here because it would keep an immutable borrow of
            // the list for the whole loop, which wouldn't allow the mutable
            // borrow of `self` in `add_backlink`
            let len = self.nodes[insert_idx].layers[lyr].len();
            for i in 0..len {
                // only borrow here, copying the value and ending the borrow before `add_backlink`
                let fw_link = self.nodes[insert_idx].layers[lyr][i];
                let backlink = Link {
                    node_index: insert_idx,
                    distance: fw_link.distance,
                };
                self.add_backlink(fw_link.node_index, backlink, lyr, select_ctx);
            }

            ep = self.nodes[insert_idx].layers[lyr]
                .iter()
                .min()
                .unwrap_or_else(|| {
                    panic!("ERROR: no neighbours found while inserting vec at layer={lyr}, an empty array was returned by select_neighbors (insert)")
                })
                .node_index;
        }

        if insert_lyr > self.max_layer {
            self.max_layer = insert_lyr;
            self.entry_point = insert_idx;
        }
    }

    fn search_layer_with_context<'a>(
        &self,
        q: &[f32; D],
        ep: usize,
        lyr: usize,
        ef: usize,
        ctx: &'a mut SearchContext,
    ) -> &'a [Link] {
        assert!(ef > 0, "ef must be > 0");
        assert!(lyr <= self.max_layer, "layer not initialized",);
        assert!(ep < self.data.len(), "entry point out of bounds",);
        assert!(
            lyr < self.nodes[ep].layers.len(),
            "entry point does not exist in this layer"
        );
        ctx.clear();
        let frontier = &mut ctx.frontier;
        let best = &mut ctx.best;
        self.epoch.set(self.epoch.get() + 1);

        let ep_link = Link {
            node_index: ep,
            distance: self.distance(q, &self.data[ep]),
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
                let dist = self.distance(q, &self.data[neigh.node_index]);
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

    fn select_neighbors(
        &self,
        qv: &[f32; D],
        lyr: usize,
        candidates: &[Link],
        extend: bool,
        keep_pruned: bool,
        ctx: &mut SelectContext,
    ) -> Vec<Link> {
        assert!(lyr <= self.max_layer, "layer not initialized",);

        ctx.clear();
        self.epoch.set(self.epoch.get() + 1);
        let pq = &mut ctx.pq;
        let discarded = &mut ctx.discarded;
        let best = &mut ctx.best;
        let max_connections = self.max_connections(lyr);

        for (node, link) in candidates
            .iter()
            .map(|link| (&self.nodes[link.node_index], link))
        {
            if node.epoch == self.epoch {
                continue;
            }
            node.epoch.set(self.epoch.get());
            pq.push(Reverse(*link));

            if extend {
                let neighs = &node.layers[lyr];
                for (neigh, vec, idx) in neighs.iter().map(|link| {
                    (
                        &self.nodes[link.node_index],
                        &self.data[link.node_index],
                        link.node_index,
                    )
                }) {
                    if neigh.epoch == self.epoch {
                        continue;
                    }
                    neigh.epoch.set(self.epoch.get());
                    pq.push(Reverse(Link {
                        node_index: idx,
                        distance: self.distance(qv, vec),
                    }));
                }
            }
        }

        self.avoid_epoch_overflow();
        // no pruning required
        if pq.len() <= max_connections {
            return ctx.consume_pq();
        }

        while let Some((vec, idx)) = pq
            .pop()
            .map(|c| (&self.data[c.0.node_index], c.0.node_index))
            && best.len() < max_connections
        {
            let mut diverse = true;
            let c_to_q = self.distance(qv, vec);
            for other in best.iter().map(|link| &self.data[link.node_index]) {
                let c_to_other = self.distance(vec, other);
                if c_to_q >= c_to_other {
                    diverse = false;
                    break;
                }
            }

            if diverse {
                best.push(Link {
                    node_index: idx,
                    distance: c_to_q,
                });
            } else if keep_pruned {
                discarded.push(Reverse(Link {
                    node_index: idx,
                    distance: c_to_q,
                }));
            }
        }

        if keep_pruned {
            while let Some(Reverse(link)) = discarded.pop()
                && best.len() < max_connections
            {
                best.push(link);
            }
        }

        ctx.consume_best()
    }

    fn add_backlink(&mut self, at: usize, link: Link, lyr: usize, ctx: &mut SelectContext) {
        assert!(lyr <= self.max_layer, "layer not initialized",);
        assert!(at < self.data.len(), "backlink base index out of bounds",);
        assert!(
            link.node_index < self.data.len(),
            "backlink connection index out of bounds"
        );
        assert!(
            lyr < self.nodes[at].layers.len(),
            "node does not exist in this layer"
        );
        assert!(link.node_index != at, "can't link node to itself");

        let max_connections = self.max_connections(lyr);
        let links = &mut self.nodes[at].layers[lyr];

        links.push(link);
        if links.len() > max_connections {
            let candidates = std::mem::take(links);
            let new_links =
                self.select_neighbors(&self.data[at], lyr, &candidates, false, false, ctx);
            self.nodes[at].layers[lyr] = new_links;
        }
    }

    #[inline(always)]
    fn max_connections(&self, lyr: usize) -> usize {
        if lyr == 0 { self.M0 } else { self.M }
    }

    #[inline(always)]
    fn distance(&self, a: &[f32; D], b: &[f32; D]) -> f32 {
        let distance = self.dist.distance(a, b);
        debug_assert!(distance.is_finite(), "distance must be finite");
        debug_assert!(distance >= 0.0, "distance must be non-negative");
        distance
    }

    #[inline(always)]
    fn random_layer(&mut self) -> usize {
        let x: f64 = Open01.sample(&mut self.rng);
        (-x.ln() * self.ml).floor() as usize
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

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D> + Default,
{
    #[allow(non_snake_case)]
    pub fn new_default(M: usize) -> Self {
        Self::new(M, 2 * M, 128, 32, DS::default())
    }
}

impl<const D: usize, DS> HnswSearcher<D> for Hnsw<D, DS>
where
    DS: Distance<D>,
{
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

        let mut ep = self.entry_point;
        for lyr in (1..=self.max_layer).rev() {
            ep = self
                .search_layer_with_context(q, ep, lyr, 1, ctx)
                .first()
                .unwrap_or_else(|| {
                    panic!("ERROR: search_layer@{lyr} returned an empty array (search)")
                })
                .node_index;
        }

        let results = self.search_layer_with_context(q, ep, 0, self.ef_search.max(k), ctx);
        // take k best from final layer search
        results[..k.min(results.len())]
            .iter()
            .map(|l| (l.node_index, l.distance))
            .collect()
    }

    fn memory_usage_bytes(&self) -> usize {
        size_of::<Self>()
            + self.data.capacity() * size_of::<[f32; D]>()
            + nodes_heap_usage_bytes(&self.nodes)
    }
}
