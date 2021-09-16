//use nalgebra::{SMatrix, RealField};
use ndarray::{Array2, ArrayView, Ix1, s};
use num::{Float};
use array_macro::array;
use std::collections::HashMap;
use crate::model_checking::utils::Hash;

const ROW: usize = 1;

/// Generic Markov Decision Process
///
/// S - State type
/// A - Action Type
/// P - Row vector of transition matrix whose size is known at compile time
/// L - Labelling type
pub trait MDP<T: Float, A, const R: usize> {
    /// Returns the initial state of the MDP
    fn initial_state(&self) -> usize;
    /// Returns the state space for the MDP
    fn get_statespace(&self) -> [usize; R];
    /// Returns a transition for a given state and action
    fn get_transition(&self, state: usize, action: usize) -> ArrayView<T, Ix1>;
}

#[derive(Clone)]
pub struct Agent<T: Float, A, const ACT: usize, const R: usize> {
    /// initial state is a reference to a state in the state space
    pub init_state: usize,
    /// State space of the MDP
    pub state_space: [State<A>; R],
    /// A mapping of the available actions for a given state
    pub state_action_map: HashMap<usize, Vec<usize>>,
    /// Actions
    pub actions: [A; ACT],
    /// The transition matrix of the MDP
    pub transition_matrix: [Array2<T>; ACT],
    /// The reward structure of the MDP
    pub rewards: Array2<T>
}

#[derive(Debug, Clone, Default)]
pub struct State<A> {
    /// A reference to the position in the state space
    pub s: usize,
    /// A generic word type, could be string, u32, usize etc
    pub w: A
}

// ---------------------------------------------------------------------------------
//              Generics of a static agent (known prior to compile time)
// ---------------------------------------------------------------------------------

impl<T: Float, A, const ACT: usize, const R: usize> MDP<T, A, R> for Agent<T, A, ACT, R> {
    fn initial_state(&self) -> usize {
        self.state_space[self.init_state].s
    }

    fn get_statespace(&self) -> [usize; R] {
        todo!()
    }

    fn get_transition(&self, s: usize, a: usize) -> ArrayView<T, Ix1> {
        self.transition_matrix[a].slice(s![s, ..])
    }
}

impl<T, A, const ACT: usize, const R: usize> Agent<T, A, ACT, R>
where A: Copy + Default, T: Clone + std::marker::Copy + Default + num_traits::identities::Zero + Float {
    fn state_space(state_labels: &HashMap<usize, A>) -> [State<A>; R] {
        let mut state_space: [State<A>; R] = array![x => State{s: x , w: A::default() }; R];
        for s in 0..R {
            match state_labels.get(&s) {
                None => {}
                Some(x) => {
                    state_space[s] = State {
                        s,
                        w: *x
                    }
                }
            }
        }
        state_space
    }

    pub fn new<'b>(init_state: usize, actions: &[A; ACT], state_act_map: &[(usize, Vec<usize>)], transition_matrix: &[&Array2<T>],
                   labels: &'b[(usize, A)], rewards: &Array2<T>) -> Agent<T, A, ACT, R> {
        let labels = labels.to_map();
        let state_space: [State<A>; R] = Agent::<T, A, ACT, R>::state_space(&labels);
        let action_map = state_act_map.to_map();
        let shape = (transition_matrix[0].shape()[0], transition_matrix[0].shape()[1]);
        let mut transitions = array![Array2::<T>::zeros(shape); ACT];
        for (i, t) in transition_matrix.iter().enumerate() {
            transitions[i] = t.view().to_owned();
        }
        let actions: [A; ACT] = actions.clone();
        Agent {
            init_state,
            state_space,
            state_action_map: action_map,
            actions,
            transition_matrix: transitions,
            rewards: rewards.clone()
        }
    }
}





