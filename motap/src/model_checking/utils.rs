use std::collections::HashMap;

pub trait Hash<A,B> {
    fn to_map(&self) -> HashMap<A, B>;
}

impl<U, T> Hash<T, U> for [(T, U)] where
    T: Clone + std::hash::Hash + std::cmp::Eq, U: std::clone::Clone {
    fn to_map(&self) -> HashMap<T, U> {
        self.iter().cloned().collect()
    }
}

