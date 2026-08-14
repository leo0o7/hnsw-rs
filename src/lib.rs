use rand::{distr::Open01, prelude::*};

use crate::{
    context::{InsertContext, SearchContext, SelectContext},
    link::Link,
    node::{Node, nodes_heap_usage_bytes},
};
use std::{cmp::Reverse, mem::size_of};

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
    pub(crate) entry_point: usize,
    pub(crate) data: Vec<[f32; D]>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) max_layer: usize,
    ml: f64,
    seed: u64,
    rng: StdRng,
    dist: DS,
}

pub trait HnswSearcher<const D: usize> {
    fn search_context(&self) -> SearchContext {
        SearchContext::default(self.len())
    }

    fn search(&self, q: &[f32; D], k: usize) -> Vec<(usize, f32)> {
        self.search_with_ef(q, k, 32)
    }

    fn search_with_ef(&self, q: &[f32; D], k: usize, ef_search: usize) -> Vec<(usize, f32)> {
        assert!(ef_search > 0, "ef_search must be > 0");
        let mut ctx = SearchContext::with_capacity(self.len(), ef_search);
        self.search_with_context(q, k, ef_search, &mut ctx)
    }

    fn search_with_context(
        &self,
        q: &[f32; D],
        k: usize,
        ef_search: usize,
        ctx: &mut SearchContext,
    ) -> Vec<(usize, f32)>;

    fn memory_usage_bytes(&self) -> usize;

    fn len(&self) -> usize;
}

// CREATE
impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D>,
{
    #[allow(non_snake_case)]
    pub fn new_seeded(M: usize, M0: usize, ef_construction: usize, seed: u64, dist: DS) -> Self {
        assert!(M > 1, "M must be > 1");
        assert!(M0 > 0, "M0 must be > 0");
        assert!(ef_construction > 0, "ef_construction must be > 0");

        let ml = 1.0 / (M as f64).ln();
        Self {
            M,
            M0,
            ef_construction,
            entry_point: 0,
            data: Vec::new(),
            nodes: Vec::new(),
            max_layer: 0,
            ml,
            seed,
            rng: StdRng::seed_from_u64(seed),
            dist,
        }
    }

    #[allow(non_snake_case)]
    pub fn new(M: usize, M0: usize, ef_construction: usize, dist: DS) -> Self {
        let seed = rand::rng().next_u64();
        Self::new_seeded(M, M0, ef_construction, seed, dist)
    }
}

// INSERT
impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D>,
{
    pub fn insert_context(&self) -> InsertContext {
        InsertContext {
            select_ctx: SelectContext::init(self.M0, self.len()),
            search_ctx: SearchContext::with_capacity(self.len(), self.ef_construction),
        }
    }

    pub fn insert(&mut self, vec: [f32; D]) {
        let mut ctx = self.insert_context();
        self.insert_with_context(vec, &mut ctx);
    }

    pub fn insert_with_context(&mut self, vec: [f32; D], ctx: &mut InsertContext) {
        let (insert_idx, insert_lyr) = self.init_node(vec);

        if insert_idx == 0 {
            self.entry_point = 0;
            self.max_layer = insert_lyr;
            return;
        }

        let search_ctx = &mut ctx.search_ctx;
        let select_ctx = &mut ctx.select_ctx;
        let mut ep = self.funnel_to_lyr_ep(vec, search_ctx, insert_lyr);
        for lyr in (0..=insert_lyr.min(self.max_layer)).rev() {
            ep = self.connect_lyr(vec, insert_idx, search_ctx, select_ctx, ep, lyr);
        }

        self.update_entry_point_if_required(insert_idx, insert_lyr);
    }

    fn connect_lyr(
        &mut self,
        vec: [f32; D],
        insert_idx: usize,
        search_ctx: &mut SearchContext,
        select_ctx: &mut SelectContext,
        ep: usize,
        lyr: usize,
    ) -> usize {
        let candidates =
            self.search_layer_with_context(&vec, ep, lyr, self.ef_construction, search_ctx);

        let selected = self.select_neighbors(&vec, lyr, candidates, false, false, select_ctx);
        let next_ep = selected
            .first()
            .expect("neighbor selection returned no nodes")
            .node_index;
        debug_assert!(
            &selected[0]
                == selected
                    .iter()
                    .min()
                    .expect("neighbor selection returned no nodes"),
            "next_ep doesn't have the distance list"
        );

        self.nodes[insert_idx].layers[lyr] = selected;
        self.publish_node(insert_idx, select_ctx, lyr);

        next_ep
    }

    fn funnel_to_lyr_ep(
        &mut self,
        vec: [f32; D],
        search_ctx: &mut SearchContext,
        insert_lyr: usize,
    ) -> usize {
        let mut ep = self.entry_point;
        for lyr in ((insert_lyr + 1)..=self.max_layer).rev() {
            ep = self
                .search_layer_with_context(&vec, ep, lyr, 1, search_ctx)
                .first()
                .unwrap_or_else(|| {
                    panic!("ERROR: search_layer@{lyr} returned an empty array (insert)")
                })
                .node_index;
        }
        ep
    }

    fn publish_node(&mut self, insert_idx: usize, select_ctx: &mut SelectContext, lyr: usize) {
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
    }

    fn init_node(&mut self, vec: [f32; D]) -> (usize, usize) {
        let insert_idx = self.data.len();
        let insert_lyr = self.random_layer();
        self.data.push(vec);
        self.nodes.push(Node {
            layers: Vec::with_capacity(insert_lyr + 1),
        });
        for lyr in 0..=insert_lyr {
            let max_connections = self.max_connections(lyr);
            self.nodes[insert_idx]
                .layers
                .push(Vec::with_capacity(max_connections));
        }
        (insert_idx, insert_lyr)
    }

    // assumes the graph is empty at creation
    fn preallocate_nodes(&mut self, vecs: Vec<[f32; D]>) -> Vec<(usize, usize)> {
        assert!(self.is_empty(), "can only preallocate when graph is empty");

        let mut nodes = Vec::with_capacity(vecs.len());
        for vec in vecs {
            let (insert_idx, insert_lyr) = self.init_node(vec);
            nodes.push((insert_idx, insert_lyr));
            self.update_entry_point_if_required(insert_idx, insert_lyr);
        }

        nodes
    }

    fn update_entry_point_if_required(&mut self, insert_idx: usize, insert_lyr: usize) {
        if insert_lyr > self.max_layer {
            self.max_layer = insert_lyr;
            self.entry_point = insert_idx;
        }
    }
}

// SEARCH
impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D>,
{
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
        let epoch = &mut ctx.visited;
        epoch.advance_epoch();
        let frontier = &mut ctx.frontier;
        let best = &mut ctx.best;

        let ep_link = Link {
            node_index: ep,
            distance: self.distance(q, &self.data[ep]),
        };
        frontier.push(Reverse(ep_link));
        best.push(ep_link);
        epoch.mark_visited(ep);

        while let Some(Reverse(candidate)) = frontier.pop() {
            let furthest_dist = best.peek().map_or(f32::INFINITY, |l| l.distance);
            if candidate.distance > furthest_dist {
                break;
            }
            for neigh in self.nodes[candidate.node_index].layers[lyr].iter() {
                if epoch.is_visited(neigh.node_index) {
                    continue;
                }
                epoch.mark_visited(neigh.node_index);
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

        ctx.consume_best()
    }
}
impl<const D: usize, DS> HnswSearcher<D> for Hnsw<D, DS>
where
    DS: Distance<D>,
{
    fn search_with_context(
        &self,
        q: &[f32; D],
        k: usize,
        ef_search: usize,
        ctx: &mut SearchContext,
    ) -> Vec<(usize, f32)> {
        assert!(ef_search > 0, "ef_search must be > 0");
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

        let results = self.search_layer_with_context(q, ep, 0, ef_search.max(k), ctx);
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

    fn len(&self) -> usize {
        self.data.len()
    }
}

// MISC
impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D>,
{
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
        let epoch = &mut ctx.visited;
        epoch.advance_epoch();
        let pq = &mut ctx.pq;
        let discarded = &mut ctx.discarded;
        let best = &mut ctx.best;
        let max_connections = self.max_connections(lyr);

        for (node, link, idx) in candidates
            .iter()
            .map(|link| (&self.nodes[link.node_index], link, link.node_index))
        {
            if epoch.is_visited(idx) {
                continue;
            }
            epoch.mark_visited(idx);
            pq.push(Reverse(*link));

            if extend {
                let neighs = &node.layers[lyr];
                for (vec, idx) in neighs
                    .iter()
                    .map(|link| (&self.data[link.node_index], link.node_index))
                {
                    if epoch.is_visited(idx) {
                        continue;
                    }
                    epoch.mark_visited(idx);
                    pq.push(Reverse(Link {
                        node_index: idx,
                        distance: self.distance(qv, vec),
                    }));
                }
            }
        }

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
        Self::new(M, 2 * M, 128, DS::default())
    }
}
