use ndarray::{ArrayView, Ix2, Array2};
use itertools::Itertools;
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
    fn state_space(&self) -> &[ProductState];
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

trait ModifyAgent {
    fn modify_transition_matrix(&self);

    fn modify_state_space(&self);

    fn rewards_model(&self);

    fn assign_rejecting(&self, acc: &[usize], graph: &Graph<usize, T>);
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
    pub s: Vec<ProductState>, // todo this might have to be dynamic, i.e. a vector and then, and then we can expose a slice to this vector in the SCPM
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
pub struct ProductState {
    pub s: i32,
    pub q: i32,
    pub ix: usize
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
        let transition_matrix = ProductAgent::<T, A, ACT, R1>::transition_matrix(mdp, task, &state_space[..]);
        let product_rewards = ProductAgent::<T, A, ACT, R1>::rewards_model(mdp, &state_space[..]);
        let graph = ProductAgent::<T, A, ACT, R1>::construct_abstract_graph(&state_space[..], &transition_matrix[..])?;
        let acc = ProductAgent::<T, A, ACT, R1>::accepting_state(&state_space[..], task, initial_state, &graph);
        let rej: Vec<usize> = ProductAgent::<T, A, ACT, R1>::construct_rejecting(&state_space[..], &acc[..], &graph).into_iter().collect();
        let (state_action_mapping, actions) = ProductAgent::<T, A, ACT, R1>::actions(&mdp, &state_space[..]);


        // We technically haven't constructed the product MDP at this point and therefore we can carry out modifcations here

        Ok(ProductAgent {
            i: initial_state,
            s: state_space,
            p: transition_matrix,
            state_hash: state_mapping,
            action_map: state_action_mapping,
            actions,
            r: product_rewards,
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
        -> (Vec<ProductState>, HashMap<(usize, usize), usize>) {
        let mut states = vec![ProductState::default(); R1];
        let mut state_hash: HashMap<(usize, usize), usize> = HashMap::new();
        let cp = mdp_states.iter().cartesian_product(dfa_states.iter());
        for (k, (s, q)) in cp.into_iter().enumerate() {
            states[k] = ProductState { s: s.s as i32, q: *q as i32, ix: k};
            state_hash.insert((s.s, *q), k);
        }
        (states, state_hash)
    }

    fn transition_matrix<const R2: usize, const R3: usize>
    (mdp: &Agent<T, A, ACT, R2>, task: &GenericTask<A, R3>, state_space: &[ProductState])
        -> [Array2<T>; ACT]
        where Agent<T, A, ACT, R2>: MDP<T, A, R2> {
        let mut p: [Array2<T>; ACT] = array![Array2::<T>::zeros((R1, R1)); ACT];
        // todo 
        for a in 0..ACT {
            for i in 0..R1 {
                for j in 0..R1 {
                    let mdp_p = &mdp.transition_matrix[a][[state_space[i].s, state_space[j].s]];
                    if *mdp_p > NumCast::from(0.0).unwrap() {
                        if state_space[j].q == task.transition(state_space[i].q as usize, mdp.state_space[state_space[j].s].w) {
                            p[a][(i,j)] = *mdp_p;
                        }
                    }
                }
            }
        }
        p
    }

    fn actions<const R2: usize>(mdp: &Agent<T, A, ACT, R2>, state_space: &[ProductState])
        -> (HashMap<usize, Vec<usize>>, [A; ACT]) {
        let mut action_hash: HashMap<usize, Vec<usize>> = HashMap::new();
        for s in state_space.iter() {
            action_hash.insert(s.ix, mdp.state_action_map.get(&s.s).unwrap().to_vec());
        }
        (action_hash, mdp.actions.clone())
    }

    /// Accepting must be reachable from initial, otherwise it will never be accepting.
    /// An empty error will return a warning
    fn accepting_state<const R3: usize>
    (state_space: &[ProductState], task: &GenericTask<A, R3>, initial: usize, graph: &Graph<usize, T>) -> Vec<usize> {
        let mut acc: Vec<usize> = Vec::new();
        let node_ix: Vec<_> = graph.node_indices().into_iter().collect();
        for s in state_space.iter() {
            if task.accepting.iter().any(|x| *x == s.q) {
                //println!("Accepting state found: {:?}", s);
                if petgraph::algo::has_path_connecting(&graph, node_ix[initial], node_ix[s.ix], None) {
                    acc.push(s.ix);
                }
            }
        }
        acc
    }

    fn construct_abstract_graph(state_space: &[ProductState], transitions: &[Array2<T>]) -> Result<Graph<usize, T, Directed>, Box<dyn std::error::Error>> {
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

    fn state_space(&self) -> &[ProductState] {
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

impl<T, A, const ACT: usize, const R1: usize> ModifyAgent for ProductAgent<T, A, ACT, R1> {
    fn modify_transition_matrix(&self) {
        todo!()
    }

    fn modify_state_space(&self) {
        todo!()
    }

    fn rewards_model(&self) {
        let mut r: Array2<T> = Array2::<T>::zeros((R1,ACT));
        for s in 0..R1 {
            for a in 0..ACT {
                r[[s, a]] = mdp.rewards[[mdp.state_space[state_space[s].s].s, a]];
            }
        }
        r
    }

    fn assign_rejecting(&self, acc: &[usize], graph: &Graph<usize, T>) -> HashSet<usize> {
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
}