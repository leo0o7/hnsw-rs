use std::{cmp::Reverse, collections::BinaryHeap};

use crate::link::Link;

pub struct SearchContext {
    pub(crate) frontier: BinaryHeap<Reverse<Link>>,
    pub(crate) best: BinaryHeap<Link>,
    results: Vec<Link>,
}

pub(crate) struct SelectContext {
    pub(crate) pq: BinaryHeap<Reverse<Link>>,
    pub(crate) discarded: BinaryHeap<Reverse<Link>>,
    pub(crate) best: Vec<Link>,
}

pub struct InsertContext {
    pub(crate) select_ctx: SelectContext,
    pub(crate) search_ctx: SearchContext,
}

impl SearchContext {
    pub(crate) fn with_capacity(cap: usize) -> Self {
        assert!(cap > 0, "search context capacity must be > 0");
        Self {
            frontier: BinaryHeap::new(),
            best: BinaryHeap::with_capacity(cap),
            results: Vec::with_capacity(cap),
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

impl Default for SearchContext {
    fn default() -> Self {
        Self::with_capacity(32)
    }
}

impl SelectContext {
    pub(crate) fn init(max_connections: usize) -> Self {
        Self {
            pq: BinaryHeap::new(),
            discarded: BinaryHeap::new(),
            best: Vec::with_capacity(max_connections),
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
