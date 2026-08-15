use rand::{distr::Open01, prelude::*};

use crate::{
    context::{InsertContext, SearchContext, SelectContext},
    link::Link,
    node::{Node, nodes_heap_usage_bytes},
};
use std::{
    cmp::Reverse,
    mem::size_of,
    sync::{Mutex, RwLock, RwLockReadGuard},
};

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
    pub(crate) storage: RwLock<Storage<D>>,
    pub(crate) entry: RwLock<(usize, usize)>,
    ml: f64,
    seed: u64,
    rng: Mutex<StdRng>,
    dist: DS,
}

#[derive(Debug)]
pub struct Storage<const D: usize> {
    pub(crate) data: Vec<[f32; D]>,
    pub(crate) nodes: Vec<Node>,
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
            entry: RwLock::new((0, 0)),
            storage: RwLock::new(Storage {
                data: Vec::new(),
                nodes: Vec::new(),
            }),
            ml,
            seed,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
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
    DS: Distance<D> + Send + Sync,
{
    pub fn insert_context(&self) -> InsertContext {
        InsertContext {
            select_ctx: SelectContext::init(self.M0, self.len()),
            search_ctx: SearchContext::with_capacity(self.len(), self.ef_construction),
        }
    }

    pub fn insert(&self, vec: [f32; D]) -> usize {
        let mut ctx = self.insert_context();
        self.insert_with_context(vec, &mut ctx)
    }

    pub fn build_parallel(&mut self, vecs: &[[f32; D]]) {
        assert!(
            self.is_empty(),
            "parallel construction is only supported on an empty graph"
        );
        let mut nodes = self.preallocate_nodes(vecs);
        // entry is already inserted
        {
            let entry = self.entry.read().unwrap();
            nodes.swap_remove(entry.0);
        }

        let nthreads = std::thread::available_parallelism()
            .expect("unable to get number available of threads")
            .get();
        let chunk_sz = nodes.len().div_ceil(nthreads);

        let index = &*self;
        std::thread::scope(|s| {
            for chunk in nodes.chunks(chunk_sz) {
                s.spawn(move || {
                    let mut thread_ctx = index.insert_context();
                    for (idx, lyr) in chunk {
                        index.insert_preallocated(*idx, *lyr, &mut thread_ctx);
                    }
                });
            }
        });
    }

    pub fn extend_parallel(&self, vecs: &[[f32; D]]) -> Vec<usize> {
        let mut internal_id = vec![0; vecs.len()];
        internal_id[0] = self.insert(vecs[0]);
        let vecs = &vecs[1..];

        let nthreads = std::thread::available_parallelism()
            .expect("unable to get number available of threads")
            .get();
        let chunk_sz = vecs.len().div_ceil(nthreads);
        std::thread::scope(|s| {
            for (chunk, interal_ids) in vecs.chunks(chunk_sz).zip(internal_id.chunks_mut(chunk_sz))
            {
                s.spawn(move || {
                    let mut thread_ctx = self.insert_context();
                    for (vec, out_id) in chunk.iter().zip(interal_ids) {
                        let id = self.insert_with_context(*vec, &mut thread_ctx);
                        *out_id = id;
                    }
                });
            }
        });

        internal_id
    }

    pub fn insert_with_context(&self, vec: [f32; D], ctx: &mut InsertContext) -> usize {
        let (node, max_lyr) = self.new_node();

        self.prepare_node(vec, &node, max_lyr, ctx);
        let idx = self.commit(vec, node);
        self.publish(idx, max_lyr, &mut ctx.select_ctx);

        idx
    }

    fn insert_preallocated(&self, idx: usize, max_lyr: usize, ctx: &mut InsertContext) -> usize {
        let storage = self.storage.read().unwrap();
        let node = &storage.nodes[idx];
        let vec = storage.data[idx];

        self.prepare_node_with_storage(&storage, vec, node, max_lyr, ctx);
        self.publish(idx, max_lyr, &mut ctx.select_ctx);
        idx
    }

    fn prepare_node(&self, vec: [f32; D], node: &Node, max_lyr: usize, ctx: &mut InsertContext) {
        let storage = self.storage.read().unwrap();
        if storage.data.is_empty() {
            return;
        }
        self.prepare_node_with_storage(&storage, vec, node, max_lyr, ctx);
    }
    // assumes non empty storage
    fn prepare_node_with_storage(
        &self,
        storage: &RwLockReadGuard<'_, Storage<D>>,
        vec: [f32; D],
        node: &Node,
        max_lyr: usize,
        ctx: &mut InsertContext,
    ) {
        let search_ctx = &mut ctx.search_ctx;
        let select_ctx = &mut ctx.select_ctx;
        let (mut ep, max_layer) = *self.entry.read().unwrap();
        for lyr in ((max_lyr + 1)..=max_layer).rev() {
            ep = self
                .search_layer_with_context(storage, &vec, ep, lyr, 1, search_ctx)
                .first()
                .unwrap_or_else(|| {
                    panic!("ERROR: search_layer@{lyr} returned an empty array (insert)")
                })
                .node_index;
        }

        for lyr in (0..=max_lyr.min(max_layer)).rev() {
            let candidates = self.search_layer_with_context(
                storage,
                &vec,
                ep,
                lyr,
                self.ef_construction,
                search_ctx,
            );
            let selected =
                self.select_neighbors(storage, &vec, lyr, candidates, false, false, select_ctx);

            let next_ep = selected
                .first()
                .expect("neighbor selection returned no nodes")
                .node_index;
            *node.layers[lyr].write().unwrap() = selected;
            ep = next_ep
        }
    }

    fn commit(&self, vec: [f32; D], node: Node) -> usize {
        let mut storage = self.storage.write().unwrap();
        let insert_idx = storage.data.len();

        storage.data.push(vec);
        storage.nodes.push(node);
        insert_idx
    }

    fn publish(&self, idx: usize, max_lyr: usize, select_ctx: &mut SelectContext) {
        let storage = self.storage.read().unwrap();
        for lyr in 0..=max_lyr {
            let links: Vec<Link> = {
                let links = storage.nodes[idx].layers[lyr].read().unwrap();
                links.iter().copied().collect()
            };
            for fw_link in links {
                let backlink = Link {
                    node_index: idx,
                    distance: fw_link.distance,
                };
                self.add_backlink(&storage, fw_link.node_index, backlink, lyr, select_ctx);
            }
        }

        self.update_entry_point_if_required(idx, max_lyr);
    }

    fn new_node(&self) -> (Node, usize) {
        let max_lyr = self.random_layer();
        let node = Node {
            layers: (0..max_lyr + 1)
                .map(|lyr| {
                    let max_connections = self.max_connections(lyr);
                    RwLock::new(Vec::with_capacity(max_connections))
                })
                .collect(),
        };

        (node, max_lyr)
    }

    fn preallocate_nodes(&self, vecs: &[[f32; D]]) -> Vec<(usize, usize)> {
        let mut storage = self.storage.write().unwrap();
        let mut out = Vec::new();
        for vec in vecs {
            let (node, max_lyr) = self.new_node();
            let idx = storage.data.len();
            storage.nodes.push(node);
            storage.data.push(*vec);

            out.push((idx, max_lyr));
        }

        out
    }

    fn update_entry_point_if_required(&self, insert_idx: usize, insert_lyr: usize) {
        let mut entry = self.entry.write().unwrap();
        if insert_lyr > entry.1 {
            entry.0 = insert_idx;
            entry.1 = insert_lyr;
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
        storage: &Storage<D>,
        q: &[f32; D],
        ep: usize,
        lyr: usize,
        ef: usize,
        ctx: &'a mut SearchContext,
    ) -> &'a [Link] {
        assert!(ef > 0, "ef must be > 0");
        assert!(lyr <= self.entry.read().unwrap().1, "layer not initialized",);
        assert!(ep < storage.data.len(), "entry point out of bounds",);
        assert!(
            lyr < storage.nodes[ep].layers.len(),
            "entry point does not exist in this layer"
        );

        ctx.clear();
        let epoch = &mut ctx.visited;
        epoch.advance_epoch();
        let frontier = &mut ctx.frontier;
        let best = &mut ctx.best;

        let ep_link = Link {
            node_index: ep,
            distance: self.distance(q, &storage.data[ep]),
        };
        frontier.push(Reverse(ep_link));
        best.push(ep_link);
        epoch.mark_visited(ep);

        while let Some(Reverse(candidate)) = frontier.pop() {
            let furthest_dist = best.peek().map_or(f32::INFINITY, |l| l.distance);
            if candidate.distance > furthest_dist {
                break;
            }
            for neigh in storage.nodes[candidate.node_index].layers[lyr]
                .read()
                .unwrap()
                .iter()
            {
                if epoch.is_visited(neigh.node_index) {
                    continue;
                }
                epoch.mark_visited(neigh.node_index);
                let dist = self.distance(q, &storage.data[neigh.node_index]);
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
        if self.is_empty() {
            return Vec::new();
        }

        let storage = self.storage.read().unwrap();
        let (mut ep, max_layer) = *self.entry.read().unwrap();
        for lyr in (1..=max_layer).rev() {
            ep = self
                .search_layer_with_context(&storage, q, ep, lyr, 1, ctx)
                .first()
                .unwrap_or_else(|| {
                    panic!("ERROR: search_layer@{lyr} returned an empty array (search)")
                })
                .node_index;
        }

        let results = self.search_layer_with_context(&storage, q, ep, 0, ef_search.max(k), ctx);
        // take k best from final layer search
        results[..k.min(results.len())]
            .iter()
            .map(|l| (l.node_index, l.distance))
            .collect()
    }

    fn memory_usage_bytes(&self) -> usize {
        let storage = self.storage.read().unwrap();
        size_of::<Self>()
            + storage.data.capacity() * size_of::<[f32; D]>()
            + nodes_heap_usage_bytes(&storage.nodes)
    }

    fn len(&self) -> usize {
        self.storage.read().unwrap().data.len()
    }
}

// MISC
impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D>,
{
    #[allow(clippy::too_many_arguments)]
    fn select_neighbors(
        &self,
        storage: &Storage<D>,
        qv: &[f32; D],
        lyr: usize,
        candidates: &[Link],
        extend: bool,
        keep_pruned: bool,
        ctx: &mut SelectContext,
    ) -> Vec<Link> {
        assert!(lyr <= self.entry.read().unwrap().1, "layer not initialized",);
        ctx.clear();
        let epoch = &mut ctx.visited;
        epoch.advance_epoch();
        let pq = &mut ctx.pq;
        let discarded = &mut ctx.discarded;
        let best = &mut ctx.best;
        let max_connections = self.max_connections(lyr);

        for (node, link, idx) in candidates
            .iter()
            .map(|link| (&storage.nodes[link.node_index], link, link.node_index))
        {
            if epoch.is_visited(idx) {
                continue;
            }
            epoch.mark_visited(idx);
            pq.push(Reverse(*link));

            if extend {
                let neighs = &node.layers[lyr];
                for (vec, idx) in neighs
                    .read()
                    .unwrap()
                    .iter()
                    .map(|link| (&storage.data[link.node_index], link.node_index))
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
            .map(|c| (&storage.data[c.0.node_index], c.0.node_index))
            && best.len() < max_connections
        {
            let mut diverse = true;
            let c_to_q = self.distance(qv, vec);
            for other in best.iter().map(|link| &storage.data[link.node_index]) {
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

    fn add_backlink(
        &self,
        storage: &RwLockReadGuard<'_, Storage<D>>,
        at: usize,
        link: Link,
        lyr: usize,
        ctx: &mut SelectContext,
    ) {
        assert!(lyr <= self.entry.read().unwrap().1, "layer not initialized",);
        assert!(at < storage.data.len(), "backlink base index out of bounds",);
        assert!(
            link.node_index < storage.data.len(),
            "backlink connection index out of bounds"
        );
        assert!(
            lyr < storage.nodes[at].layers.len(),
            "node does not exist in this layer"
        );
        assert!(link.node_index != at, "can't link node to itself");

        let max_connections = self.max_connections(lyr);
        let mut links = storage.nodes[at].layers[lyr].write().unwrap();

        links.push(link);
        if links.len() > max_connections {
            let candidates = std::mem::take(&mut *links);
            let new_links = self.select_neighbors(
                storage,
                &storage.data[at],
                lyr,
                &candidates,
                false,
                false,
                ctx,
            );
            *links = new_links;
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
    fn random_layer(&self) -> usize {
        let x: f64 = Open01.sample(&mut self.rng.lock().unwrap());
        (-x.ln() * self.ml).floor() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.storage.read().unwrap().data.is_empty()
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
