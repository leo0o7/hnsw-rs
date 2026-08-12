use serde::{Deserialize, Serialize};

use crate::link::Link;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub layers: Vec<Vec<Link>>,
}

pub(crate) fn nodes_heap_usage_bytes(nodes: &Vec<Node>) -> usize {
    let mut bytes = nodes.capacity() * size_of::<Node>();
    for node in nodes {
        bytes += node.layers.capacity() * size_of::<Vec<Link>>();
        for layer in &node.layers {
            bytes += layer.capacity() * size_of::<Link>();
        }
    }
    bytes
}
