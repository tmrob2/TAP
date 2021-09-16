use std::collections::HashMap;
use ndarray::{Array2};
use ndarray_csv::Array2Reader;
use csv::{ReaderBuilder};
use std::path::Path;
use array_macro::array;
use std::fmt::Debug;
use crate::model_checking::utils::Hash;

/// Generic DFA
///
/// Q - state type of the DFA
/// W - words in an alphabet
pub trait DFA<A> {
    /// return the initial state of the DFA
    fn initial_state(&self) -> usize;
    /// A deterministic transition in a DFA, given a word and a state there is only one
    /// possible state returned
    fn transition(&self, state: usize, word: A) -> usize;
    /// Given a state determine if it is accepting
    fn is_accepting(&self, state: usize) -> bool;
}

/// Task formulation: R - number of states, C - Number of letters in alphabet,
/// Sigma
pub struct GenericTask<A, const R: usize> {
    pub initial_state: usize,
    pub states: [usize; R],
    pub transitions: Array2<usize>,
    pub alphabet: HashMap<A, usize>,
    pub accepting: Vec<usize>
}

// ---------------------------------------------------------------------------------
//             Setting up the generics of a task specific to this model
// ---------------------------------------------------------------------------------

impl<A, const R: usize> GenericTask<A, R>
    where A: std::hash::Hash + std::cmp::Eq + std::clone::Clone {
    fn transitions(fname: &str, alphabet_size: usize) -> Result<Array2<usize>, Box<dyn std::error::Error>> {
        let path = Path::new("/home/tmrob2/Rust/motap/data/");
        let csv_file = path.join(fname);
        let mut reader = ReaderBuilder::new().has_headers(false).from_path(csv_file)?;
        // { R } x { C } type multiplication; R interpreted as a constant
        let array: Array2<usize> = reader.deserialize_array2((R, { alphabet_size }))?;
        Ok(array)
    }

    pub fn new(fname: &str, initial_state: usize, alphabet: &[(A, usize)], accepting: Vec<usize>)
        -> Result<GenericTask<A, R>, Box<dyn std::error::Error>> {
        let transitions: Array2<usize> = GenericTask::<A, R>::transitions(fname, alphabet.len())?;
        let alphabet_map = alphabet.to_map();
        Ok(GenericTask {
            initial_state,
            states: array![x => x; R],
            transitions,
            alphabet: alphabet_map,
            accepting
        })
    }
}

impl<A, const R: usize> DFA<A> for GenericTask<A, R>
    where A: std::hash::Hash + std::cmp::Eq + Debug {
    fn initial_state(&self) -> usize {
        self.initial_state
    }

    fn transition(&self, state_ix: usize, word: A) -> usize
    {
        let word_ix = self.alphabet.get(&word).unwrap();
        self.transitions[(state_ix, *word_ix)]
    }

    fn is_accepting(&self, state: usize) -> bool {
        self.accepting.iter().any(|x| *x == state)
    }
}