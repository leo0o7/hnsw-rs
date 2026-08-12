use std::{cmp::Reverse, collections::BinaryHeap};

use crate::link::Link;

pub struct SearchContext {
    pub(crate) frontier: BinaryHeap<Reverse<Link>>,
    pub(crate) best: BinaryHeap<Link>,
    pub(crate) visited: VisitedSet,
    results: Vec<Link>,
}

pub(crate) struct SelectContext {
    pub(crate) pq: BinaryHeap<Reverse<Link>>,
    pub(crate) discarded: BinaryHeap<Reverse<Link>>,
    pub(crate) best: Vec<Link>,
    pub(crate) visited: VisitedSet,
}

pub struct InsertContext {
    pub(crate) select_ctx: SelectContext,
    pub(crate) search_ctx: SearchContext,
}

impl SearchContext {
    pub(crate) fn with_capacity(nodes_cnt: usize, cap: usize) -> Self {
        assert!(cap > 0, "search context capacity must be > 0");
        Self {
            frontier: BinaryHeap::new(),
            best: BinaryHeap::with_capacity(cap),
            results: Vec::with_capacity(cap),
            visited: VisitedSet::new(nodes_cnt),
        }
    }

    pub(crate) fn default(nodes_cnt: usize) -> Self {
        Self {
            frontier: BinaryHeap::new(),
            best: BinaryHeap::new(),
            results: Vec::with_capacity(32),
            visited: VisitedSet::new(nodes_cnt),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.frontier.clear();
        self.best.clear();
        self.results.clear();
    }

    pub(crate) fn consume_best(&mut self) -> &[Link] {
        while let Some(p) = self.best.pop() {
            self.results.push(p);
        }
        self.results.reverse();
        &self.results
    }
}

impl SelectContext {
    pub(crate) fn init(nodes_cnt: usize, max_connections: usize) -> Self {
        Self {
            pq: BinaryHeap::new(),
            discarded: BinaryHeap::new(),
            best: Vec::with_capacity(max_connections),
            visited: VisitedSet::new(nodes_cnt),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.pq.clear();
        self.discarded.clear();
        self.best.clear();
    }

    pub(crate) fn consume_pq(&mut self) -> Vec<Link> {
        let mut out = Vec::with_capacity(self.pq.len());
        while let Some(Reverse(p)) = self.pq.pop() {
            out.push(p);
        }
        out
    }

    pub(crate) fn consume_best(&mut self) -> Vec<Link> {
        self.best.to_vec()
    }
}

pub(crate) struct VisitedSet {
    node_epoch: Vec<usize>,
    epoch: usize,
}

impl VisitedSet {
    fn new(nodes_cnt: usize) -> Self {
        Self {
            node_epoch: vec![0; nodes_cnt],
            epoch: 0,
        }
    }

    pub(crate) fn is_visited(&mut self, node: usize) -> bool {
        self.grow_if_needed(node);
        self.node_epoch[node] == self.epoch
    }

    pub(crate) fn mark_visited(&mut self, node: usize) {
        self.grow_if_needed(node);
        self.node_epoch[node] = self.epoch;
    }

    pub(crate) fn advance_epoch(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.node_epoch.fill(0);
            self.epoch += 1;
        }
    }

    // TODO: avoid this
    fn grow_if_needed(&mut self, node: usize) {
        if node >= self.node_epoch.len() {
            self.node_epoch.resize(node + 1, 0);
        }
    }
}
