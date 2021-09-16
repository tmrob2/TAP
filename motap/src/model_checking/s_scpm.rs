//use nalgebra::{SMatrix, RealField, Scalar, MatrixSlice, SVector, Dim, MatrixSlice1x2, U1, Const, RowSVector};
extern crate openblas_src;
use ndarray::{arr1, Array1, Array2, ArrayView, Ix1, Ix2, s, LinalgScalar};
use ndarray_stats::{QuantileExt, DeviationExt};
use std::collections::{HashMap, HashSet};
use crate::model_checking::product_mdp::{ProductAgentInner, Dims, PMDPIdent};
use array_macro::array;
use num_traits::{Zero, One};
use std::marker::Copy;
use std::ops::{Div, Sub, Mul, Neg, Add};
use std::fmt::Debug;
use std::error::Error;
use num::{Float, NumCast};
use ndarray_stats::errors::MinMaxError;

/// Generic Sequential Consecutive Product MDP (SCPM)
///
/// Type S - State space type
/// Type T - [f32, f64] type of static transition matrix
/// const ACT - number of actions in the SCPM
/// const N - number of sub-product MDPs
/// const R - number of states in SCPM
///

const ROW: usize = 1;

pub struct SCPM<T, A, const S: usize, const ACT: usize, const N: usize> where T: std::fmt::Display {
    pub s: [State; S],
    pub p: [Array2<T>; ACT],
    pub r: [Array2<T>; ACT],
    task_automaton_state_ranges: HashMap<PMDPIdent, MatrixRange>,
    initial_states_mapping: HashMap<PMDPIdent, usize>,
    action_label_mapping: HashMap<A, usize>
}

#[derive(Default, Clone, Debug)]
pub struct State {
    pub s: i32,
    pub q: i32,
    pub action_set: Vec<usize>,
    pub switch_exists: bool,
    pub agent: usize,
    pub task: usize,
    pub ix: usize,
    map_pmdp_ix: usize
}

#[derive(Debug)]
struct MatrixRange {
    lower: usize,
    upper: usize
}

/// N - dimension of the problem agents + tasks
impl<T, A, const S: usize, const ACT: usize, const N: usize> SCPM<T, A, S, ACT, N>
    where T: Clone + num_traits::Zero + Float + Mul<Output=T> + Neg<Output=T> + std::fmt::Debug + std::fmt::Display,
          A: Default + Clone + Copy + Eq + std::hash::Hash + std::fmt::Debug {
    fn state_space(&mut self, pmdps: &[Box<dyn ProductAgentInner<T, A>>], tasks: usize, agents: usize) {
        let mut state_ctr: usize = 0;
        for task in 0..tasks {
            for agent in 0..agents {
                let ord_pmdp = pmdps.iter()
                    .find(|x|
                        x.identifiers().task == task &&
                            x.identifiers().agent == agent).unwrap();
                let pmdp_state_space = ord_pmdp.state_space();
                for s in pmdp_state_space.iter() {
                    if ord_pmdp.expose_initial() == s.ix {
                        self.initial_states_mapping.insert(ord_pmdp.identifiers(), state_ctr);
                    }
                    let switch = if ord_pmdp.non_reachable(s.ix) && s.s == ord_pmdp.mdp_init_state() {
                        true
                    } else {
                        false
                    };
                    let state_actions: Vec<A> = ord_pmdp.action_set(s.ix).iter().map(|x| ord_pmdp.expose_act_labels()[*x]).collect();
                    let state_act_ix: Vec<usize> = state_actions.iter().map(|x| *self.action_label_mapping.get(x).unwrap()).collect();
                    self.s[state_ctr] = State {
                        s: s.s,
                        q: s.q,
                        action_set: state_act_ix,
                        switch_exists: switch,
                        agent,
                        task,
                        ix: state_ctr,
                        map_pmdp_ix: s.ix
                    };
                    state_ctr += 1;
                }
            }
        }
    }

    /// Constructs a series of rewards matrices for given actions
    ///
    /// Act: a ->
    ///
    /// |    |a_1 cost | a_2 cost | ... | a_n cost |   t_1 reward | t_2 reward  | ...  |t_m reward |
    /// |----|---------| -------- | ----| -------- | ------------ | ----------- | ---- | --------- |
    /// |s_0 | c_11    |    c_12  |     | c_1n     |rho_1(s_0,a)  | rho_2(s_0,a)|      |rho_m(s_0,a)|
    /// |s_1 |         |          |     |          |              |             |      |           |
    /// |... |         |          |     |          |              |             |      |           |
    /// |s_n |         |          |     |          |              |             |      |           |
    /// --------------------------------------------------------------------------------------------
    ///
    /// The value of c_ij is inherited from the product MDP
    /// The value of the task reward is evaluated according to a function of whether the sub
    /// product mdp task DFA is in an accepting state, rewarded by a 1.0 and 0.0 otherwise.
    fn reward_model(&mut self, pmdps: &[Box<dyn ProductAgentInner<T, A>>], agents: usize, tasks: usize) {
        for a in 0..ACT {
            for pmdp in pmdps.iter() {
                let idents = pmdp.identifiers();
                let rng = self.task_automaton_state_ranges.get(&idents).unwrap();
                let mut pmdp_s: usize = 0;
                for s in rng.lower..rng.upper + 1 {
                    if idents.agent == agents - 1 && idents.task == tasks - 1 {
                        // No more rewards are accumulated once reachability is confirmed
                        if !pmdp.non_reachable(pmdp_s) {
                            self.r[a][[s, idents.agent]] = T::from(-1.0).unwrap() * pmdp.rewards(a, pmdp_s);
                        }
                    } else {
                        self.r[a][[s, idents.agent]] = T::from(-1.0).unwrap() * pmdp.rewards(a, pmdp_s);
                    }

                    if pmdp.accepting(pmdp_s) {
                        self.r[a][[s, idents.agent]] = T::from(0.0).unwrap();
                        self.r[a][[s, idents.task + agents]] = T::from(1000.0).unwrap();
                    }
                    pmdp_s += 1;
                }
            }
        }
    }

    fn index_mapping(&mut self, pmdps: &[Box<dyn ProductAgentInner<T, A>>], tasks: usize, agents: usize) {
        let mut prev_upper_bound: usize = 0;
        // Ordering should be the same as the user input
        for task in 0..tasks{
            for agent in 0..agents {
                let pmdp = pmdps.iter()
                    .find(|x|
                        x.identifiers().task == task &&
                            x.identifiers().agent == agent).unwrap();
                let dimensions: Dims = pmdp.dimensions(0);
                let upper_bnd = prev_upper_bound + dimensions.rows - 1;
                let lower_bnd = prev_upper_bound;
                prev_upper_bound = upper_bnd + 1;
                self.task_automaton_state_ranges.insert(pmdp.identifiers(), MatrixRange{lower:lower_bnd, upper: upper_bnd});
            }
        }
    }

    fn get_actions(&mut self, pmdps: &[Box<dyn ProductAgentInner<T, A>>]) {
        let mut action_set: HashMap<A, usize> = HashMap::new();
        let mut act_ctr: usize = 0;
        for pmdp in pmdps.iter() {
            let acts: &[A] = pmdp.expose_act_labels();
            //println!("act: {:?}", acts);
            for a in acts.iter() {
                if !action_set.contains_key(a) {
                    action_set.insert(*a, act_ctr);
                    act_ctr += 1;
                }
            }
            //println!("action set: {:?}", action_set);
        }
        self.action_label_mapping = action_set;
        //println!("{:?}", self.action_label_mapping);
    }

    /// Constructs a new SCPM model. Requires a set of product MDP models, the number of tasks
    /// and the number of agents included in a problem
    pub fn new(&mut self, pmdps: &[Box<dyn ProductAgentInner<T, A>>], tasks: usize, agents: usize, switch_label: A) {
        self.get_actions(pmdps);
        self.index_mapping(pmdps, tasks, agents);
        self.state_space(pmdps, tasks, agents);
        self.reward_model(pmdps, agents, tasks);
    }

    pub fn default() -> SCPM<T, A, S, ACT, N> {
        SCPM {
            s: array![State::default(); S],
            p: array![Array2::<T>::zeros((S, S)); ACT],
            r: array![Array2::<T>::zeros((S, N)); ACT],
            task_automaton_state_ranges: Default::default(),
            initial_states_mapping: Default::default(),
            action_label_mapping: Default::default()
        }
    }
}

trait LinAlgMethods<T, A> where T: LinalgScalar + num_traits::Zero + std::marker::Copy + One + std::ops::Sub + std::ops::Div  {
    fn dot_product_transitions_xval(&self, s: usize, a: usize, pmdp: &Box<dyn ProductAgentInner<T, A>>, x: &ArrayView<T, Ix1>) -> T;

    fn dot_product_rewards(&self, s: usize, a: usize, w: &ArrayView<T, Ix1>) -> T;

    //fn abs_diff_max(&self, x: Array2<T>) -> Result<&T, MinMaxError>;
}

impl<T, A, const S: usize, const ACT: usize, const N: usize> LinAlgMethods<T, A> for SCPM<T, A, S, ACT, N>
    where T: Float + LinalgScalar + num_traits::Zero + std::marker::Copy + One + std::ops::Sub + std::ops::Div + Debug + std::fmt::Display {
    fn dot_product_transitions_xval(&self, s: usize, a: usize, pmdp: &Box<dyn ProductAgentInner<T, A>>, x: &ArrayView<T, Ix1>) -> T {
        pmdp.expose_matrix(a).slice(s![s, ..]).dot(x)
    }

    fn dot_product_rewards(&self, s: usize, a: usize, w: &ArrayView<T, Ix1>) -> T {
        self.r[a].slice(s![s, ..]).dot(w)
    }

    /*fn abs_diff_max(&self, x: Array2<T>) -> Result<&T, MinMaxError> {
        x.mapv(T::abs).max()
    }*/
}

trait SchedulerSynthesis<T, A> {
    fn mult_obj_sheduler_synthesis(&self, target: &[T], eps: &T, tasks: usize, agents: usize, pmdpms: &[Box<dyn ProductAgentInner<T, A>>]) -> Result<(), Box<dyn std::error::Error>>;

    fn exp_tot_cost(&self, w: &ArrayView<T, Ix1>, eps: &T, agents: usize, tasks: usize, pmdps: &[Box<dyn ProductAgentInner<T, A>>]) -> Result<(), Box<dyn std::error::Error>>;
}

pub trait ValueIteration<T, A> {
    fn run(&self, target: &[T], eps: &T, agents: usize, tasks: usize, pmdps: &[Box<dyn ProductAgentInner<T, A>>]) -> Result<(), Box<dyn std::error::Error>>;
}

impl<T, A, const S: usize, const ACT: usize, const N: usize> SchedulerSynthesis<T, A> for SCPM<T, A, S, ACT, N>
where T: Float + Debug + LinalgScalar + Zero + Copy + Clone + One + Sub + Div + PartialOrd + std::fmt::Display {
    fn mult_obj_sheduler_synthesis(&self, target: &[T], eps: &T, tasks: usize, agents: usize, pmdps: &[Box<dyn ProductAgentInner<T, A>>]) -> Result<(), Box<dyn Error>> {
        let mut _hullset: Vec<Array1<T>> = Vec::new();
        let mut _mu_sol: Vec<Array1<T>> = Vec::new();

        let extreme_points: Array2<T> = Array2::<T>::eye(N);

        for r in 0..N {
            let w: ArrayView<T, Ix1> = extreme_points.slice(s![r, ..]);
            self.exp_tot_cost(&w, eps, agents, tasks, pmdps);
        }
        Ok(())
    }

    fn exp_tot_cost(&self, w: &ArrayView<T, Ix1>, eps: &T, agents: usize, tasks: usize, pmdps: &[Box<dyn ProductAgentInner<T, A>>]) -> Result<(), Box<dyn std::error::Error>> {
        let mut mu: [usize; S] = array![0; S];
        let mut r: [T; N] = array![T::from(0.0).unwrap(); N];
        let mut x = arr1(&array![T::from(0.0).unwrap(); S]);
        let mut y = arr1(&array![T::from(0.0).unwrap(); S]);
        let mut ymat = Array2::<T>::zeros((N, S));
        let mut xmat = Array2::<T>::zeros((N, S));
        for j in (0..tasks).rev() {
            // next pmdp: For switch transitions. The next pmdp only depends on the task, and we don't want to waste
            // unnecessary searches for pmdps in the agent loop, therefore it is called earlier than the task-agent pmdp.
            let next_init_ix: Option<usize> = if j < tasks - 1 {
                let z = pmdps.iter().find(|k| k.identifiers().task == j + 1 && k.identifiers().agent == 0);
                Some(*self.initial_states_mapping.get(&z.unwrap().identifiers()).unwrap())
            } else {
                None
            };
            for i in (0..agents).rev() {
                let pmdp = pmdps.iter().find(|k| k.identifiers().task == j && k.identifiers().agent == i).unwrap();
                let rng: &MatrixRange = self.task_automaton_state_ranges.get(&pmdp.identifiers()).unwrap();
                let mut epsilon: T = T::from(1.0).unwrap();
                while epsilon > *eps {
                    for s in self.s[rng.lower..rng.upper + 1].iter() {
                        let mut max_value: T = T::neg_infinity();
                        let mut chosen_action: usize = 0;
                        let mut value: T = T::from(0.0).unwrap();
                        //println!("action set: {:?}", s.action_set);
                        for a in s.action_set.iter() {
                            //println!("a: {:?}", a);
                            value = self.dot_product_rewards(s.ix, *a, w)
                                + self.dot_product_transitions_xval(s.map_pmdp_ix, *a, pmdp, &x.slice(s![rng.lower..rng.upper + 1]));
                            if s.switch_exists && j < tasks - 1 {
                                println!("state: ({},{}), action: {}, reward: {:?}, pdotx: {:?}, y: {:?}, x: {:?}", s.s, s.q, a, self.r[chosen_action].slice(s![s.ix, ..]), self.dot_product_transitions_xval(s.map_pmdp_ix, chosen_action, pmdp, &x.slice(s![rng.lower..rng.upper + 1])), y[s.ix], x[s.ix]);
                                println!("x: {:.1}", &x.slice(s![rng.lower..rng.upper + 1]));
                            }
                            if value > max_value {
                                max_value = value;
                                chosen_action = *a;
                            }
                        }
                        // determine if this state has a switch transition
                        if s.switch_exists {
                            match next_init_ix {
                                None => {}
                                Some(ix) => {
                                    if j < tasks - 1 {
                                        println!("state: ({},{}), max_value: {}, {:?}", s.s, s.q, max_value, x[ix]);
                                    }
                                    value = x[ix];
                                    if value > max_value {
                                        max_value = value;
                                        chosen_action = ACT;
                                    }
                                }
                            }
                        }
                        y[s.ix] = max_value;
                        mu[s.ix] = chosen_action;
                    }
                    epsilon = *(&y - &x).max()?;
                    //println!("epsilon: {:?}", epsilon);
                    x.assign(&y);
                    println!("x: {:.1}", x);
                }
                return Ok(());
                epsilon = T::from(1.0).unwrap();

                /*
                while epsilon > *eps {
                    for s in self.s[rng.lower..rng.upper + 1].iter() {
                        let chosen_action: usize = mu[s.ix];
                        for k in 0..N {
                            if chosen_action < ACT {
                                ymat[[k, s.ix]] = self.r[chosen_action][[s.ix, k]]
                                    + self.dot_product_transitions_xval(s.map_pmdp_ix, chosen_action, pmdp, &xmat.slice(s![k, rng.lower..rng.upper + 1]));
                                //println!("k: {}, state: ({},{}), chosen action: {}, reward: {:?}, pdotx: {:?}, y: {:?}, x: {:?}", k, s.s, s.q, chosen_action, self.r[chosen_action][[s.ix, k]], self.dot_product_transitions_xval(s.map_pmdp_ix, chosen_action, pmdp, &xmat.slice(s![k, rng.lower..rng.upper + 1])), ymat[[k, s.ix]], xmat[[k, s.ix]]);
                                //println!("{:.1}", xmat.slice(s![k, ..]));
                            } else {
                                match next_init_ix {
                                    None => {}
                                    Some(ix) => {
                                        ymat[[k, s.ix]] = xmat[[k + 1, ix]];
                                    }
                                }
                            }
                        }
                    }
                    epsilon = (&ymat - &xmat).mapv(T::abs).max()?.clone();
                    println!("{:?}", epsilon);
                    xmat.assign(&ymat);
                }

                 */
            }
        }
        for k in 0..N {
            r[k] = ymat[[k, 0]];
        }
        println!("r: {:?}", r);
        let output: Vec<((usize, usize), usize)> = self.s.iter().map(|x| ((x.s, x.q), mu[x.ix])).collect();
        println!("mu: {:?}", mu);
        Ok(())
    }
}

impl<T, A, const S: usize, const ACT:usize, const N: usize> ValueIteration<T, A> for SCPM<T, A, S, ACT, N>
where T: Float + std::fmt::Debug + LinalgScalar + Zero + Copy + Clone + One + Sub + Div + PartialOrd + std::fmt::Display {
    fn run(&self, target: &[T], eps: &T, agents: usize, tasks: usize, pmdps: &[Box<dyn ProductAgentInner<T, A>>])
        -> Result<(), Box<dyn std::error::Error>> {
        let w = arr1(&[T::from(0.0).unwrap(), T::from(0.5).unwrap(), T::from(0.5).unwrap()]); //
        self.exp_tot_cost(&w.view(), eps, agents, tasks, pmdps)?;
        //self.mult_obj_sheduler_synthesis(target, eps, tasks, agents, pmdps);
        Ok(())
    }
}




