use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::SMatrix;
use std::collections::HashMap;
use criterion::measurement::WallTime;
use std::borrow::Cow;

type Matrix5x5f64 = SMatrix<f64, 2, 2>;

pub fn from_vector(v: &[&Matrix5x5f64], ix: usize) -> &Matrix5x5f64 {
    v[ix]
}

pub fn vector_selection(c: &mut Criterion) -> &mut Criterion<WallTime> {
    let m1 = Matrix5x5f64::new(
        1.0, 0.0,
        0.0, 1.0
    );
    let m2 = Matrix5x5f64::new(
        1.0, 0.0,
        0.0, 1.0
    );
    let v: Vec<&Matrix5x5f64> = vec![&m1, &m2];
    c.bench_function("matrix sel vector index", |b| b.iter(|| from_vector(&v[..], 0)))
}

