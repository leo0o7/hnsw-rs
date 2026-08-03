use std::{cell::Cell, error::Error, fs::File, io::BufWriter, path::Path};

use rand::{
    SeedableRng,
    distr::{Distribution, Open01},
    rngs::StdRng,
};
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    ser::{SerializeSeq, SerializeStruct},
};

use crate::{Hnsw, dist::Distance, node::Node};

struct FlatF32<'a, const D: usize>(&'a [[f32; D]]);
impl<'a, const D: usize> From<&'a [[f32; D]]> for FlatF32<'a, D> {
    fn from(value: &'a [[f32; D]]) -> Self {
        Self(value)
    }
}

#[allow(non_snake_case)]
#[derive(Deserialize)]
struct SerializedHnsw<DS> {
    M: usize,
    M0: usize,
    ef_construction: usize,
    entry_point: usize,
    data: Vec<Vec<f32>>,
    nodes: Vec<Node>,
    max_layer: usize,
    ml: f64,
    seed: u64,
    dist: DS,
}

impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D> + Serialize,
{
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let f = File::create(path)?;
        let w = BufWriter::new(f);
        bincode2::serialize_into(w, self)?;
        Ok(())
    }
}

impl<const D: usize, DS> Hnsw<D, DS>
where
    DS: Distance<D> + for<'de> Deserialize<'de>,
{
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let b = std::fs::read(path)?;
        Ok(bincode2::deserialize(&b)?)
    }
}

impl<'a, const D: usize> Serialize for FlatF32<'a, D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for v in self.0 {
            seq.serialize_element(&v[..])?;
        }
        seq.end()
    }
}

impl<const D: usize, DS> Serialize for Hnsw<D, DS>
where
    DS: Distance<D> + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Hnsw", 9)?;
        state.serialize_field("M", &(self.M as u64))?;
        state.serialize_field("M0", &(self.M0 as u64))?;
        state.serialize_field("ef_construction", &(self.ef_construction as u64))?;
        state.serialize_field("entry_point", &(self.entry_point as u64))?;
        state.serialize_field("data", &FlatF32::from(self.data.as_slice()))?;
        state.serialize_field("nodes", &self.nodes)?;
        state.serialize_field("max_layer", &(self.max_layer as u64))?;
        state.serialize_field("ml", &self.ml)?;
        state.serialize_field("seed", &self.seed)?;
        state.serialize_field("dist", &self.dist)?;
        state.end()
    }
}

impl<'de, const D: usize, DS> Deserialize<'de> for Hnsw<D, DS>
where
    DS: Distance<D> + Deserialize<'de>,
{
    fn deserialize<DE>(deserializer: DE) -> Result<Self, DE::Error>
    where
        DE: serde::Deserializer<'de>,
    {
        let disk = SerializedHnsw::<DS>::deserialize(deserializer)?;

        let mut data = Vec::with_capacity(disk.data.len());
        for vec in disk.data {
            if vec.len() != D {
                return Err(serde::de::Error::custom(format!(
                    "invalid vector dimensions, expected {D}, got {}",
                    vec.len()
                )));
            }
            data.push(vec.try_into().expect("impossible"));
        }

        // advance rng
        let mut rng = StdRng::seed_from_u64(disk.seed);
        for _ in 0..data.len() {
            let _: f64 = Open01.sample(&mut rng);
        }

        Ok(Self {
            M: disk.M,
            M0: disk.M0,
            ef_construction: disk.ef_construction,
            entry_point: disk.entry_point,
            data,
            nodes: disk.nodes,
            max_layer: disk.max_layer,
            epoch: Cell::new(0),
            ml: disk.ml,
            seed: disk.seed,
            rng,
            dist: disk.dist,
        })
    }
}

pub(crate) fn serialize_epoch_as_zero<S>(
    _epoch: &Cell<usize>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_u64(0u64)
}

pub(crate) fn deserialize_epoch_as_zero<'de, D>(deserializer: D) -> Result<Cell<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    let _ = usize::deserialize(deserializer)?;
    Ok(Cell::new(0))
}
