use ndarray::{ArrayView, Ix2, Array2};
use itertools::{Itertools, Product};
use super::s_mdp::{MDP, Agent, State};
use crate::model_checking::s_automaton::{GenericTask, DFA};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use petgraph::{Graph, dot::Dot, Directed, graph::NodeIndex};
use std::fs::File;
use std::io::Write;
use std::fmt::{Display, Debug};
use array_macro::array;
use petgraph::graph::node_index;
use num::{Float, NumCast};

// ---------------------------------------------------------------------------------
//                             Generic Product MDP model
// ---------------------------------------------------------------------------------

pub trait Analysis<T> {
    fn construct_str_graph(&self, fname: &str) -> Result<(), Box<dyn std::error::Error>>;
}

/// Because the Product MDPs are of different sizes, we require a trait to expose the
/// attributes of a Product MDP, which we can use in a boxed trait object.
pub trait ProductAgentInner<T, A> {
    /// Exposes the transition matrix for a given action a: usize
    fn expose_matrix(&self, a: usize) -> ArrayView<T, Ix2>;
    /// Exposes the initial state index of the PMDP
    fn expose_initial(&self) -> usize;
    /// Exposes the state space of a PMDP
    fn state_space(&self) -> &[ModProductState];
    /// Exposes the identifies 'agent number, task number' for a given PMDP. This function allows
    /// ignoring the ordering of the PMDPs in construction of the SCPM.
    fn identifiers(&self) -> PMDPIdent;
    /// Exposes the dimensions of the transition model for a given action a: usize
    fn dimensions(&self, a: usize) -> Dims;
    /// Exposes the rewards model of the Product MDP
    fn rewards(&self, a: usize, s: usize) -> T;
    /// The states under which the Product MDP is accepting for the given task
    fn accepting(&self, s: usize) -> bool;
    /// Any state from which an accepting state cannot be reached
    fn non_reachable(&self, s: usize) -> bool;
    /// Available actions
    fn action_set(&self, s: usize) -> &[usize];
    /// mdp initial state
    fn mdp_init_state(&self) -> usize;
    /// Exposes the action labels for a given type A
    fn expose_act_labels(&self) -> &[A];
}

/// Struct for exposing the dimensions of a transition SMatrix
pub struct Dims {
    pub rows: usize,
    pub cols: usize
}

/// Struct for exposing the identification of a Product MDP
#[derive(Eq, Hash, PartialEq)]
pub struct PMDPIdent {
    pub task: usize,
    pub agent: usize
}

/// Generic Product MDP
pub struct ProductAgent<T, A, const ACT: usize, const R1: usize> {
    pub i: usize,
    pub s: Vec<ModProductState>, // todo this might have to be dynamic, i.e. a vector and then, and then we can expose a slice to this vector in the SCPM
    pub p: [Array2<T>; ACT],
    state_hash: HashMap<(usize, usize), usize>,
    action_map: HashMap<usize, Vec<usize>>,
    actions: [A; ACT],
    pub r: Array2<T>,
    pub agent: usize,
    agent_init: usize,
    pub task: usize,
    pub acc: Vec<usize>,
    pub rej: Vec<usize>
}

#[derive(Copy, Default, Clone, Debug)]
struct ProductState {
    s: usize,
    q: usize,
    ix: usize
}

#[derive(Copy, Default, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ModProductState {
    pub s: i32,
    pub q: i32,
    pub ix: usize
}

#[derive(Clone, Eq, Hash, PartialEq, Debug)]
struct ModInstruction {
    prev_trans: Transition,
    future_trans: [Transition; 2],
    state: ModProductState,
    a: usize
}

#[derive(Clone, Eq, Hash, PartialEq, Default, Debug)]
struct Transition {
    from: usize,
    to: usize
}

struct FutureTransition;

impl FutureTransition {
    /// A future transition is a modification that is a path from i -> k -> j
    fn apply(i: usize, j: usize, k: usize) -> [Transition; 2] {
        [Transition { from: i, to: k }, Transition { from: k, to: j }]
    }
}

impl ModInstruction {
    fn new(i: usize, j: usize, k: usize, a: usize, state: ModProductState) -> ModInstruction {
        ModInstruction {
            prev_trans: Transition {from: i, to: j},
            future_trans: FutureTransition::apply(i,j,k),
            state,
            a
        }
    }
}

impl<T, A, const ACT: usize, const R1: usize> ProductAgent<T, A, ACT, R1>
where T: Float + PartialOrd + std::fmt::Debug + std::cmp::PartialEq
+ Copy + std::clone::Clone + num_traits::Zero + std::fmt::Display + ToString,
      A: Display + Clone + Copy + Debug + std::hash::Hash + Eq{
    pub fn new<const R2:usize, const R3: usize>
    (mdp: &Agent<T, A, ACT, R2>, task: &GenericTask<A, R3>, agent_no: usize, task_no: usize)
        -> Result<ProductAgent<T, A, ACT, R1>, Box<dyn std::error::Error>> {
        let (state_space, state_mapping): ([ProductState; R1], HashMap<(usize, usize), usize>) =
            ProductAgent::<T, A, ACT, R1>::statespace(&mdp.state_space[..], &task.states[..]);
        let initial_state = ProductAgent::<T, A, ACT, R1>::initial_state(mdp, &task, &state_space[..]);
        let (transition_matrix, modifications) = ProductAgent::<T, A, ACT, R1>::transition_matrix(mdp, task, &state_space[..]);
        let modified_transition_matrix = ProductAgent::<T, A, ACT, R1>::modify_transition_matrix(&transition_matrix[..], &modifications);
        let modified_state_space = ProductAgent::<T, A, ACT, R1>::modify_state_space(&state_space[..], &modifications);
        let rewards_model = ProductAgent::<T, A, ACT, R1>::rewards_model(&mdp, &modified_state_space[..]);
        let (state_action_mapping, actions) = ProductAgent::<T, A, ACT, R1>::actions(&mdp, &modified_state_space[..], &modifications);
        let graph = ProductAgent::<T, A, ACT, R1>::construct_abstract_graph(&modified_state_space[..], &modified_transition_matrix[..])?;
        let rej = ProductAgent::<T, A, ACT, R1>::assign_rejecting(&modified_state_space[..], &task.accepting[..], &graph).into_iter().collect();
        let acc = ProductAgent::<T, A, ACT, R1>::accepting_state(&modified_state_space[..], &task, initial_state, &graph);
        Ok(ProductAgent {
            i: initial_state,
            s: modified_state_space,
            p: modified_transition_matrix,
            state_hash: state_mapping,
            action_map: state_action_mapping,
            actions,
            r: rewards_model,
            agent: agent_no,
            agent_init: mdp.init_state,
            task: task_no,
            acc,
            rej
        })
    }

    fn initial_state<const R2: usize, const R3: usize>
    (mdp: &Agent<T, A, ACT, R2>, task: &GenericTask<A, R3>, state_space: &[ProductState])  -> usize
        where Agent<T, A, ACT, R2>: MDP<T, A, R2> {
        let init_state = state_space.iter().position(|x| x.s == mdp.init_state && x.q == task.initial_state).unwrap();
        init_state
    }

    fn statespace(mdp_states: &[State<A>], dfa_states: &[usize])
        -> ([ProductState; R1], HashMap<(usize, usize), usize>) {
        let mut states = array![ProductState::default(); R1];
        let mut state_hash: HashMap<(usize, usize), usize> = HashMap::new();
        let cp = mdp_states.iter().cartesian_product(dfa_states.iter());
        for (k, (s, q)) in cp.into_iter().enumerate() {
            states[k] = ProductState { s: s.s, q: *q, ix: k};
            state_hash.insert((s.s, *q), k);
        }
        (states, state_hash)
    }

    fn transition_matrix<const R2: usize, const R3: usize>(mdp: &Agent<T, A, ACT, R2>, task: &GenericTask<A, R3>, state_space: &[ProductState])
        -> ([Array2<T>; ACT], HashSet<ModInstruction>)
        where Agent<T, A, ACT, R2>: MDP<T, A, R2> {
        let mut p: [Array2<T>; ACT] = array![Array2::<T>::zeros((R1, R1)); ACT];
        let mut modifications: HashSet<ModInstruction> = HashSet::new();
        let mut mod_count: usize = R1;
        for a in 0..ACT {
            for i in 0..R1 {
                for j in 0..R1 {
                    let mdp_p = &mdp.transition_matrix[a][[state_space[i].s, state_space[j].s]];
                    if *mdp_p > NumCast::from(0.0).unwrap() {
                        if state_space[j].q == task.transition(state_space[i].q, mdp.state_space[state_space[j].s].w) {
                            p[a][(i,j)] = *mdp_p;
                            if !task.is_accepting(state_space[i].q) && task.is_accepting(state_space[j].q) {
                                //println!("s: ({},{}) -> s': ({},{}), accepting transition, ready for modification", state_space[i].s, state_space[i].q, state_space[j].s, state_space[j].q);
                                //println!("s: ({},{}) -> s*: ({},{}) -> s': ({},{})", state_space[i].s, state_space[i].q, mod_count, state_space[j].q, state_space[j].s, state_space[j].q);
                                modifications.insert(ModInstruction::new(i, j, mod_count, a, ModProductState {s: -1, q: state_space[j].q as i32, ix: mod_count}));
                                mod_count += 1;
                            }
                        }
                    }
                }
            }
        }
        (p, modifications)
    }
    /// To convert reachability probabilities to reachability rewards, we must modify the state space to include a one-way transition
    /// which can only be reached once and then continues to accepted states.
    fn modify_transition_matrix(transition_matrix: &[Array2<T>], modifications: &HashSet<ModInstruction>) -> [Array2<T>; ACT] {
        // Size the array
        let mod_size: usize = modifications.len();
        let base_matrix: Array2<T> = Array2::<T>::zeros((R1 + mod_size, R1 + mod_size));
        // copy over all of the elements of the previous transition matrix
        let mut mod_transition_matrix = array![base_matrix; ACT];
        for a in 0..ACT {
            for i in 0..R1 {
                for j in 0..R1 {
                    mod_transition_matrix[a][[i,j]] = transition_matrix[a][[i,j]];
                }
            }
        }
        for instr in modifications.iter() {
            // Future instruction 1: rerouting the pre accepting state to the diode state
            mod_transition_matrix[instr.a][[instr.future_trans[0].from, instr.future_trans[0].to]] =
                mod_transition_matrix[instr.a][[instr.prev_trans.from, instr.prev_trans.to]];
            // Future instruction 2: rerouting the diode state to the accepting state, will occur almost surely
            mod_transition_matrix[instr.a][[instr.future_trans[1].from, instr.future_trans[1].to]] = T::from(1.0).unwrap();
            // Deleting the previous transition
            mod_transition_matrix[instr.a][[instr.prev_trans.from, instr.prev_trans.to]] = T::from(0.0).unwrap();
        }
        mod_transition_matrix
    }

    fn modify_state_space(state_space: &[ProductState], modification: &HashSet<ModInstruction>) -> Vec<ModProductState> {
        // First copy over the original state space, and then add in the new states
        let mut mod_state_space: Vec<ModProductState> = state_space.iter().map(|x| ModProductState {
            s: x.s as i32,
            q: x.q as i32,
            ix: x.ix
        } ).collect();
        for instr in modification.iter() {
            mod_state_space.push(instr.state.clone());
        }
        mod_state_space
    }
    // todo needs to be adjusted to the modified state space
    fn rewards_model<const R2: usize>(mdp: &Agent<T, A, ACT, R2>, state_space: &[ModProductState]) -> Array2<T> {
        let mut r: Array2<T> = Array2::<T>::zeros((R1,ACT));
        for s in 0..R1 {
            for a in 0..ACT {
                if state_space[s].s >= 0 {
                    r[[s, a]] = mdp.rewards[[mdp.state_space[state_space[s].s as usize].s, a]];
                } else {
                    r[[s, a]] = T::from(0.0).unwrap();
                }
            }
        }
        r
    }
    // todo needs to be adjusted to the modified transition matrix
    fn assign_rejecting(state_space: &[ModProductState], acc: &[usize], graph: &Graph<usize, T>) -> HashSet<usize> {
        let mut rej: HashSet<usize> = HashSet::new();
        let node_ix: Vec<_> = graph.node_indices().into_iter().collect();
        for s in state_space.iter() {
            for a in acc.iter() {
                if !petgraph::algo::has_path_connecting(graph, node_ix[s.ix], node_ix[*a], None) {
                    rej.insert(s.ix);
                }
            }
        }
        rej
    }
    // todo needs to be adjusted to the modified state space
    fn actions<const R2: usize>(mdp: &Agent<T, A, ACT, R2>, state_space: &[ModProductState], modifications: &HashSet<ModInstruction>) -> (HashMap<usize, Vec<usize>>, [A; ACT]) {
        let mut action_hash: HashMap<usize, Vec<usize>> = HashMap::new();
        for s in state_space.iter() {
            if s.s >= 0 {
                action_hash.insert(s.ix, mdp.state_action_map.get(&(s.s as usize)).unwrap().to_vec());
            } else {
                action_hash.insert(s.ix, vec![modifications.iter().find(|x| x.state == *s).unwrap().a]);
            }
        }
        (action_hash, mdp.actions.clone())
    }
    // todo needs to be adjusted to the modified transition matrix
    fn construct_abstract_graph(state_space: &[ModProductState], transitions: &[Array2<T>]) -> Result<Graph<usize, T, Directed>, Box<dyn std::error::Error>> {
        let mut graph: Graph<usize, T, Directed> = Graph::new();
        for s in state_space.iter() {
            graph.add_node(s.ix);
        }
        let node_ix: Vec<_> = graph.node_indices().into_iter().collect();
        for a in 0..ACT {
            for s in state_space.iter() {
                for sprime in state_space.iter() {
                    if transitions[a][[s.ix, sprime.ix]] > NumCast::from(0.0).unwrap() {
                        graph.add_edge(node_ix[s.ix], node_ix[sprime.ix], transitions[a][[s.ix, sprime.ix]]);
                    }
                }
            }
        }
        Ok(graph)
    }
    // todo this needs to be adjusted to only accept the diode state
    fn accepting_state<const R3: usize>(state_space: &[ModProductState], task: &GenericTask<A, R3>, initial: usize, graph: &Graph<usize, T>) -> Vec<usize> {
        let mut acc: Vec<usize> = Vec::new();
        let node_ix: Vec<_> = graph.node_indices().into_iter().collect();
        for s in state_space.iter() {
            if task.accepting.iter().any(|x| *x == s.q as usize) {
                //println!("Accepting state found: {:?}", s);
                if petgraph::algo::has_path_connecting(&graph, node_ix[initial], node_ix[s.ix], None) {
                    acc.push(s.ix);
                }
            }
        }
        acc
    }
}

impl<T, A, const ACT: usize, const R1: usize> Analysis<T> for ProductAgent<T, A, ACT, R1>
    where T: Float + PartialOrd + Display +
    std::fmt::Debug + std::cmp::PartialEq + std::cmp::PartialOrd + Copy + std::clone::Clone + num_traits::identities::Zero,
          A: Display + Clone + Copy + Debug + std::hash::Hash + Eq {
    fn construct_str_graph(&self, fname: &str) -> Result<(), Box<dyn Error>> {
        let mut graph: Graph<String, String> = Graph::new();
        for state in self.s.iter() {
            graph.add_node(format!("({},{})", state.s, state.q));
        }
        for a in 0..ACT {
            for i in 0..R1 {
                for j in 0..R1 {
                    if self.p[a][[i,j]] > T::from(0.0).unwrap() {
                        let origin_index: NodeIndex = graph.node_indices()
                            .find(|x| graph[*x] == format!("({},{})", self.s[i].s, self.s[i].q)).unwrap();
                        let destination_index: NodeIndex = graph.node_indices()
                            .find(|x| graph[*x] == format!("({},{})", self.s[j].s, self.s[j].q)).unwrap();
                        let label = format!("p:{},a:{}",self.p[a][[i, j]], a);
                        graph.add_edge(origin_index, destination_index, label);
                    }
                }
            }
        }
        let dot = Dot::with_config(&graph, &[]).to_string();
        let mut file = File::create(fname)?;
        file.write_all(&dot.as_bytes())?;
        Ok(())
    }
}

impl<T, A, const ACT: usize, const R: usize> ProductAgentInner<T, A> for ProductAgent<T, A, ACT, R>
    where T: Debug + Copy + Float,
          A: Display + Clone + Copy + Debug + std::hash::Hash + Eq {
    fn expose_matrix(&self, a: usize) -> ArrayView<T, Ix2> {
        self.p[a].view()
    }

    fn expose_initial(&self) -> usize {
        self.i
    }

    fn state_space(&self) -> &[ModProductState] {
        &self.s[..]
    }

    fn identifiers(&self) -> PMDPIdent {
        PMDPIdent {
            task: self.task,
            agent: self.agent
        }
    }

    fn dimensions(&self, a: usize) -> Dims {
        let shape = self.p[a].shape();
        Dims {
            rows: shape[0],
            cols: shape[1]
        }
    }

    fn rewards(&self, a: usize, s: usize) -> T {
        self.r[[s, a]]
    }

    fn accepting(&self, s: usize) -> bool {
        self.acc.iter().any(|x| *x == s)
    }

    fn non_reachable(&self, s: usize) -> bool {
        self.rej.iter().any(|x| *x == s)
    }

    fn action_set(&self, s: usize) -> &[usize] {
        self.action_map.get(&s).unwrap()
    }

    fn mdp_init_state(&self) -> usize {
        self.agent_init
    }

    fn expose_act_labels(&self) -> &[A] {
        &self.actions[..]
    }
}