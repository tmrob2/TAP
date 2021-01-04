use std::convert::TryFrom;
use minilp::{Problem, OptimizationDirection, ComparisonOp, Variable, LinearExpr};
use std::collections::{HashSet, VecDeque,HashMap};
use rand::Rng;
//use rand::seq::{SliceRandom, IteratorRandom};
use std::hash::Hash;
//use itertools::{assert_equal, Itertools};
use ordered_float::NotNan;
use ndarray::{arr1, Array1};
extern crate petgraph;
use petgraph::graph::Graph;
use petgraph::dot::Dot;
use std::fmt;
//use std::fs::File;
//use std::io::prelude::*;
use rand::seq::SliceRandom;

//##################################################################################
//                              MDP STRUCTURES
//##################################################################################
#[derive(Debug, Clone)]
pub struct ProductTransitionPair {
    pub s: u32,
    pub p: f64,
    pub q: Vec<u32>,
    pub abstract_label: HashMap<u8, TaskProgress>,
    //pub accepting: bool,
    //pub rejecting: bool,
    pub ap: String,
    pub stoppable: bool,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq,)]
pub struct ProductStateSpace {
    pub s: u32,
    pub q: Vec<u32>,
    pub mdp_init: bool
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct TaskAction {
    pub a: i8,
    pub task: u8,
}

impl fmt::Display for TaskAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}", self.a, self.task)
    }
}

impl TaskAction {
    pub fn default() -> TaskAction {
        TaskAction {
            a: -1,
            task: 0
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProductTransition {
    pub s: u32,
    pub q: Vec<u32>,
    pub a: TaskAction,
    pub rewards: f64,
    pub s_prime: Vec<ProductTransitionPair>,
    pub abstract_label: HashMap<u8, TaskProgress>,
    // we actually gain nothing by storing which task is a just fail because construction
    // of the modified product vector relies on looking at all s' from s
    pub stoppable: bool,
    //pub accepted: bool,
    //pub rejected: bool,
    pub ap: String,
}

pub struct DFA <'a> {
    pub states: Vec<u32>,
    pub initial: u32,
    pub delta: fn(u32, &'a str) -> u32,
    pub rejected: Vec<u32>,
    pub accepted: Vec<u32>
}

pub struct MDP {
    pub states: Vec<u32>,
    pub initial: u32,
    pub transitions: Vec<Transition>,
    pub labelling: fn(u32) -> &'static str,
}

pub fn is_stoppable(abstract_label: &HashMap<u8, TaskProgress>) -> bool {
    let in_progress = abstract_label.values().any(|x| match x {
        TaskProgress::InProgress => true,
        _ => false,
    });
    if in_progress {
        false
    } else {
        true
    }
}

impl MDP {

    pub fn get_abstract_label(q: Option<&u32>, qprime: &u32, dfa: &DFA) -> TaskProgress {
        match q {
            Some(x) => {
                if dfa.accepted.contains(qprime) {
                    TaskProgress::Finished
                }
                else if dfa.rejected.contains(qprime) && dfa.rejected.contains(x) {
                    TaskProgress::Failed
                } else if dfa.rejected.contains(qprime) && !dfa.rejected.contains(x) {
                    TaskProgress::JustFailed
                } else if dfa.initial == *qprime {
                    // check if any other task is inProgress
                    TaskProgress::Initial
                } else {
                    TaskProgress::InProgress
                }
            },
            None => {
                if dfa.accepted.contains(qprime) {
                    TaskProgress::Finished
                }
                else if dfa.rejected.contains(qprime) {
                    TaskProgress::Failed
                } else if dfa.initial == *qprime {
                    TaskProgress::Initial
                } else {
                    TaskProgress::InProgress
                }
            }
        }
    }

    pub fn initial_product_mdp<'a>(&'a self, dfa: &'a DFA<'a>, empty_container: &'a mut ProductMDP<'a>) -> &'a mut ProductMDP<'a> {
        // create a new state space based on the mdp and the dfa
        let mut state_space_new: Vec<ProductStateSpace> = Vec::new();
        let mut transitions_new: Vec<ProductTransition> = Vec::new();
        //let mut state_index: u8 = 0;
        for state in self.states.iter() {
            for q in dfa.states.iter() {
                for transition in self.transitions.iter().filter(|x| x.s == *state) {
                    let mut sprimes: Vec<ProductTransitionPair> = Vec::new();
                    for sprime in transition.s_prime.iter() {
                        // name the dfa state that we are going to with the transition label
                        let qprime: u32 = (dfa.delta)(*q, (self.labelling)(sprime.s));
                        // determine if it is a self loop or not
                        let mut task_progress: HashMap<u8, TaskProgress> = HashMap::new();
                        let progress_value = MDP::get_abstract_label(Some(&q), &qprime, &dfa);
                        task_progress.insert(0, progress_value);
                        sprimes.push(ProductTransitionPair{
                            s: sprime.s,
                            p: sprime.p,
                            q: vec![qprime],
                            abstract_label: task_progress.clone(),
                            ap: format!("{}",(self.labelling)(sprime.s)),
                            stoppable: is_stoppable(&task_progress),
                        });
                    }
                    let mut state_progress: HashMap<u8, TaskProgress> = HashMap::new();
                    let state_progress_value = MDP::get_abstract_label(None, q, &dfa);
                    state_progress.insert(0, state_progress_value);
                    transitions_new.push(ProductTransition{
                        s: *state,
                        q: vec![*q],
                        a: TaskAction {a: transition.a, task: 0},
                        rewards: transition.rewards,
                        s_prime: sprimes,
                        abstract_label: state_progress.clone(),
                        stoppable: is_stoppable(&state_progress),
                        ap: format!("{}", (self.labelling)(*state)),
                    })
                }
                let mdp_init_value: bool = if *state == self.initial {true} else {false};
                state_space_new.push(ProductStateSpace{ s: *state, q: vec![*q], mdp_init: mdp_init_value});
            }
        }
        empty_container.dfa_delta.insert(0u8, &dfa.delta);
        empty_container.states = state_space_new;
        empty_container.transitions = transitions_new;
        empty_container.initial = ProductStateSpace{ s: self.initial, q: vec![dfa.initial], mdp_init: true};
        empty_container.labelling = vec![self.labelling];
        empty_container.task_counter = 1;
        empty_container
    }
}

#[derive(Debug, Copy, Clone)]
pub enum TaskProgress {
    Failed,
    JustFailed,
    Finished,
    Initial,
    InProgress
}

#[derive(Clone)]
pub struct ModifiedProductMDP<'a> {
    pub states: Vec<ProductStateSpace>,
    pub transitions: Vec<ModProductTransition>,
    pub initial: ProductStateSpace,
    pub labelling: &'a Vec<fn(u32) -> &'static str>,
    pub number: u8,
    pub task_counter: u8,
}

#[derive(Debug, Clone)]
pub struct ModProductTransitionPair {
    pub s: u32,
    pub p: f64,
    pub q: Vec<u32>,
    pub abstract_label: HashMap<u8, TaskProgress>,
    pub ap: String,
    pub state_index: usize,
    pub stoppable: bool,
}

pub fn tasks_finished(transition: &HashMap<u8, TaskProgress>) -> bool {
    let mut task_progress: Vec<bool> = Vec::new();
    for (_k, v) in transition {
        match v {
            TaskProgress::Failed => task_progress.push(true),
            TaskProgress::Finished => task_progress.push(true),
            TaskProgress::JustFailed => task_progress.push(true),
            _ => task_progress.push(false)
        }
    }
    if task_progress.iter().all(|x| *x == true) {
        true
    } else {
        false
    }
}

impl <'a> ModifiedProductMDP <'a> {

    fn mod_abstract_ap(abstract_ap: Option<&TaskProgress>) -> TaskProgress {
        match abstract_ap {
            Some(x) => match *x {
                TaskProgress::Initial => TaskProgress::InProgress,
                _ => *x
            },
            None => {println!("Found empty task progress"); TaskProgress::Initial}
        }
    }

    pub fn task_rewards(abstract_label: &HashMap<u8, TaskProgress>) -> HashMap<u8, f64> {
        let mut rewards= HashMap::new();
        for (k,v) in abstract_label.iter() {
            match v {
                TaskProgress::JustFailed => rewards.insert(*k, 1.),
                _ => rewards.insert(*k, 0.)
            };
        }
        rewards
    }

    /// A function that assigns a state vector index to its corresponding transitions for later use in the
    /// value iteration algorithms
    pub fn assign_state_index_to_transition(&mut self) -> () {
        for transition in self.transitions.iter_mut() {
            let index = self.states.iter().position(|x| x.s == transition.s && x.q == transition.q).unwrap();
            transition.state_index = index;
            for sprime in transition.s_prime.iter_mut() {
                let sprime_index = self.states.iter().position(|x| x.s == sprime.s && x.q == sprime.q).unwrap();
                sprime.state_index = sprime_index;
            }
        }
    }

    pub fn generate_mod_product(&mut self, pmdp: &'a ProductMDP) -> &'a mut ModifiedProductMDP {
        // There are three steps to generating a modified product MDP
        // 1. A new state is added for self loops, this is so that we can clearly identify if the
        //    task has begun
        // 2. A state is added for justFail processes, this is so that we can conduct reachability
        //    analysis
        // 3. All other transitions, and states need to be inherited from the base local product

        // self loops definitions for A'(s,q) subset A(s,q) such that a in A'(s,q) iff
        // P'((s,q), a, (s,q)) > 0 && P'((s,q), a, (s'q'))>0 for some (s',q') != (s,q)
        let mut unique_state_space: HashSet<ProductStateSpace> = HashSet::new();
        let mut mod_prod_transitions: Vec<ModProductTransition> = Vec::new();
        let mut counter: u32 = u32::try_from(pmdp.states.len()).unwrap();
        for transition in pmdp.transitions.iter() {
            // that is for self loop identification, there needs to be a edge from (s,q) -> (s,q)
            // and there needs to be an edge from (s,q) -> (s',q') where (s',q') != (s,q)
            let state_initial = pmdp.states.iter().filter(|x| x.s == transition.s && x.q == transition.q).next().unwrap().mdp_init;
            unique_state_space.insert(ProductStateSpace{ s: transition.s, q: transition.q.to_vec(), mdp_init: state_initial});
            let loop_cond: bool = transition.s_prime.iter().any(|x| x.s == transition.s);
            let no_loop_cond: bool = transition.s_prime.iter().any(|x| x.s != transition.s);
            let mut transition_abstract_label_self_loop: HashMap<u8, TaskProgress> = HashMap::new();
            if loop_cond == true && no_loop_cond == true {
            //if transition.self_loop == true {
                // add a new state to the state space, this state will be snew
                unique_state_space.insert(ProductStateSpace{ s: counter, q: transition.q.to_vec(), mdp_init: false });
                //println!("Self loop");
                // edit the current transition so that is goes to the new state, and then add a transition from the
                // new state to the original sprime state
                // Another thing to consider is that this must be a self loop, it cannot be anything else because it
                // satisfies the definition, therefore all of the sprimes must be edited
                let mut sprimes_current: Vec<ModProductTransitionPair> = Vec::new();
                let mut sprimes_snew: Vec<ModProductTransitionPair> = Vec::new();
                for sprime in transition.s_prime.iter(){
                    // if (s,q) != (s',q') =>  copy, and copy and edit from new state
                    //&& sprime.q != transition.q
                    if sprime.s != transition.s {
                        //println!("progress: ({}, {:?}) -> ({},{:?})", transition.s, transition.q, sprime.s, sprime.q);
                        // this is not a loop
                        sprimes_current.push(ModProductTransitionPair{
                            s: sprime.s,
                            p: sprime.p,
                            q: sprime.q.to_vec(),
                            abstract_label: sprime.abstract_label.clone(),
                            ap: "".to_string(),
                            state_index: 0,
                            stoppable: sprime.stoppable
                        }); // original this will be from snew to s'
                        sprimes_snew.push(ModProductTransitionPair{
                            s: sprime.s,
                            p: sprime.p,
                            q: sprime.q.to_vec(),
                            abstract_label: sprime.abstract_label.clone(),
                            ap: "".to_string(),
                            state_index: 0,
                            stoppable: sprime.stoppable
                        });
                        // this will be the original AP
                    } else {
                        //println!("loop: ({}, {:?}) -> ({},{:?})", transition.s, transition.q, sprime.s, sprime.q);
                        // these should only be self loops, but there may be circumstances where we have been tricked somewhere
                        // in the creation of the product MDP, something to watch out for...
                        // in the case where is it a loop, both versions of the transitions will be edited, and the abstract AP will be edited
                        let mut abstract_ap_new = sprime.abstract_label.clone();
                        let task_progress_new = ModifiedProductMDP::mod_abstract_ap(abstract_ap_new.get(&transition.a.task));
                        // we will need an extra function here to declare whether the new ap for s' is stoppable

                        abstract_ap_new.remove(&transition.a.task);
                        abstract_ap_new.insert(transition.a.task, task_progress_new);
                        transition_abstract_label_self_loop = abstract_ap_new.clone();
                        let sprime_loop_val = ModProductTransitionPair {
                            s: counter,
                            p: sprime.p,
                            q: sprime.q.to_vec(),
                            abstract_label: abstract_ap_new,
                            ap: "".to_string(), // starting to question if we need a string interpretation of the abstract ap
                            stoppable: false,
                            state_index: 0,
                        };
                        sprimes_current.push(sprime_loop_val.clone());
                        sprimes_snew.push(sprime_loop_val.clone());
                    }
                }
                // this is the current transition
                let task_rewards = ModifiedProductMDP::task_rewards(&transition.abstract_label);
                mod_prod_transitions.push(ModProductTransition{
                    s: transition.s,
                    q: transition.q.to_vec(),
                    a: TaskAction {a: transition.a.a, task: transition.a.task},
                    constraint: transition.rewards,
                    reach_rewards: task_rewards.clone(),
                    s_prime: sprimes_current,
                    abstract_label: transition.abstract_label.clone(),
                    ap: Vec::new(),
                    stoppable: transition.stoppable,
                    state_index: 0,
                });
                // this is the transitions for snew
                mod_prod_transitions.push(ModProductTransition{
                    s: counter,
                    q: transition.q.to_vec(),
                    a: TaskAction {a: transition.a.a, task: transition.a.task},
                    constraint: transition.rewards,
                    reach_rewards: task_rewards.clone(),
                    s_prime: sprimes_snew,
                    abstract_label: transition_abstract_label_self_loop,
                    ap: Vec::new(),
                    stoppable: false, //is_stoppable(&transition.abstract_label),
                    state_index: 0,
                });
                counter = counter + 1;

            }
            else if loop_cond == true && no_loop_cond == false {
                // there is only a loop to itself, and nothing else, therefore no progress is possible
                //println!("pure loop")
                //println!("Pure loop: {:?}", transition);
                let task_rewards = ModifiedProductMDP::task_rewards(&transition.abstract_label);
                let mut sprimes: Vec<ModProductTransitionPair> = Vec::new();
                for sprime in transition.s_prime.iter() {
                    sprimes.push(ModProductTransitionPair{
                        s: sprime.s,
                        p: sprime.p,
                        q: sprime.q.to_vec(),
                        abstract_label: sprime.abstract_label.clone(),
                        ap: "".to_string(),
                        state_index: 0,
                        stoppable: sprime.stoppable
                    })
                }
                mod_prod_transitions.push(ModProductTransition{
                    s: transition.s,
                    q: transition.q.to_vec(),
                    a: TaskAction {a: transition.a.a, task: transition.a.task},
                    s_prime: sprimes,
                    constraint: transition.rewards,
                    reach_rewards: task_rewards,
                    ap: vec![],
                    abstract_label: transition.abstract_label.clone(),
                    stoppable: transition.stoppable,
                    state_index: 0,
                });
            }
            else {
                //println!("Another type of transition")
                // what is the current task
                let current_task = &transition.a.task;
                // is the current AP a non justFail
                let current_state_abstract_ap: bool = match transition.abstract_label.get(current_task) {
                    Some(x) => match x {
                        TaskProgress::InProgress => true,
                        TaskProgress::Initial => true,
                        _ => false,
                    },
                    None => false
                };
                if current_state_abstract_ap == true {
                    let mut sprime_current: Vec<ModProductTransitionPair> = Vec::new();
                    //println!("Orginal: ({},{:?}) -> ({:?})", transition.s, transition.q, transition.s_prime);
                    let task_rewards = ModifiedProductMDP::task_rewards(&transition.abstract_label);
                    for sprime in transition.s_prime.iter() {
                        // if there is an s' which has a task member of justFail, then we add a new state, called s*
                        match sprime.abstract_label.get(current_task) {
                            Some(x) => match x {
                                TaskProgress::JustFailed => {
                                    //println!("justFail transition found, state: ({}, {:?}), {}, ({}, {:?}) task: {}",
                                    //         transition.s, transition.q, transition.a.a, sprime.s, sprime.q, current_task);
                                    unique_state_space.insert(ProductStateSpace{ s: counter, q: sprime.q.to_vec(), mdp_init: false });
                                    // a product transition to s*, then a product transition from s* to the original s',
                                    // which is just a copy of the original transition
                                    let mut new_abstract_label: HashMap<u8, TaskProgress> = sprime.abstract_label.clone();

                                    new_abstract_label.remove(&transition.a.task);
                                    new_abstract_label.insert(transition.a.task, TaskProgress::Failed);
                                    let sstar_sprime = ModProductTransitionPair{
                                        s: sprime.s,
                                        p: 1.,
                                        q: sprime.q.to_vec(),
                                        abstract_label: new_abstract_label,
                                        ap: "".to_string(),
                                        state_index: 0,
                                        stoppable: true
                                    };
                                    let task_rewards2 = ModifiedProductMDP::task_rewards(&sprime.abstract_label);
                                    mod_prod_transitions.push(ModProductTransition{
                                        s: counter,
                                        q: sprime.q.to_vec(),
                                        a: TaskAction {a: transition.a.a, task: transition.a.task},
                                        constraint: 0., // we say that the cost is zero to take the action tau in state s*
                                        reach_rewards: task_rewards2,
                                        s_prime: vec![sstar_sprime],
                                        ap: vec![format!("justFail{}", transition.a.task)],
                                        abstract_label: sprime.abstract_label.clone(),
                                        stoppable: true,
                                        state_index: 0,
                                    });

                                    sprime_current.push(ModProductTransitionPair{
                                        s: counter,
                                        p: sprime.p,
                                        q: sprime.q.to_vec(),
                                        abstract_label: sprime.abstract_label.clone(),
                                        ap: "".to_string(),
                                        state_index: 0,
                                        stoppable: true // bercause this is the transition to s*
                                    });
                                    //println!("Added transition: ({},{:?}) -> ({},{:?}) -> ({}, {:?})", transition.s, transition.q.to_vec(), counter, sprime.q, sprime.s, sprime.q);
                                    counter = counter + 1;
                                },
                                _ => {
                                    // this is the case where we do not see a transition to justFail, and therefore we just copy the transition
                                    sprime_current.push(ModProductTransitionPair{
                                        s: sprime.s,
                                        p: sprime.p,
                                        q: sprime.q.to_vec(),
                                        abstract_label: sprime.abstract_label.clone(),
                                        ap: "".to_string(),
                                        state_index: 0,
                                        stoppable: sprime.stoppable,
                                    });
                                }
                            },
                            None => {println!("The current task did not have an abstract label");}
                        }
                    }
                    mod_prod_transitions.push(ModProductTransition{
                        s: transition.s,
                        q: transition.q.to_vec(),
                        a: TaskAction {a: transition.a.a, task: transition.a.task},
                        constraint: transition.rewards,
                        reach_rewards: task_rewards,
                        s_prime: sprime_current,
                        ap: vec![format!("!justFail{}", transition.a.task)],
                        abstract_label: transition.abstract_label.clone(),
                        stoppable: transition.stoppable,
                        state_index: 0,
                    })

                } else {
                    // we are in an end condition and we should just copy over the transition
                    let task_rewards = ModifiedProductMDP::task_rewards(&transition.abstract_label);
                    let mut sprimes: Vec<ModProductTransitionPair> = Vec::new();
                    for sprime in transition.s_prime.iter(){
                        sprimes.push(ModProductTransitionPair{
                            s: sprime.s,
                            p: sprime.p,
                            q: sprime.q.to_vec(),
                            abstract_label: sprime.abstract_label.clone(),
                            ap: "".to_string(),
                            state_index: 0,
                            stoppable: sprime.stoppable
                        })
                    }
                    mod_prod_transitions.push(ModProductTransition{
                        s: transition.s,
                        q: transition.q.to_vec(),
                        a: TaskAction {a: transition.a.a, task: transition.a.task},
                        constraint: transition.rewards,
                        reach_rewards: task_rewards,
                        s_prime: sprimes,
                        ap: vec![],
                        abstract_label: transition.abstract_label.clone(),
                        stoppable: transition.stoppable,
                        state_index: 0,
                    })
                }
            }
        }

        self.states = unique_state_space.into_iter().collect();
        self.transitions = mod_prod_transitions;
        self.task_counter = pmdp.task_counter;
        self.initial = pmdp.initial.clone();
        self

    }

    /// This function acts as the scheduler would act and non-deterministically selects the next action
    /// of a series of actions representing tasks action sets.
    fn determine_choices(&self, s: &u32, q: &Vec<u32>) -> TraversalStateSpace {
        let mut choices: Vec<ModProductTransition> = Vec::new();
        for transition in self.transitions.iter().filter(|x| x.s == *s && x.q == *q) {
            choices.push(transition.clone());
        }
        match choices.choose(&mut rand::thread_rng()) {
            Some(x) => TraversalStateSpace {
                state: ProductStateSpace { s: x.s, q: x.q.to_vec(), mdp_init: false },
                a: TaskAction { a: x.a.a, task: x.a.task },
                abstract_label: x.abstract_label.clone(),
                stoppable: x.stoppable
            },
            None => TraversalStateSpace {
                state: ProductStateSpace { s: 0, q: vec![], mdp_init: false },
                a: TaskAction { a: 0, task: 0 },
                abstract_label: Default::default(),
                stoppable: false
            }
        }
    }

    pub fn traverse_n_steps(&self){
        //let mut step: u32 = 0;
        let mut new_state: Traversal;// = Traversal::default();
        let mut finished: bool = false;
        let mut transition_choice = self.determine_choices(&self.initial.s, &self.initial.q);
        let mut current_state = TraversalStateSpace {
            state: ProductStateSpace { s: transition_choice.state.s, q: transition_choice.state.q.to_vec(), mdp_init: false },
            a: TaskAction { a: transition_choice.a.a, task: transition_choice.a.task },
            abstract_label: transition_choice.abstract_label.clone(),
            stoppable: transition_choice.stoppable
        };
        print!("p((s:{},q{:?}) , a:{:?}, (s':{},q':{:?})): ", &self.initial.s, &self.initial.q, &current_state.a, &current_state.state.s, &current_state.state.q);
        println!("abstract label: {:?} -> {:?}", self.label(&self.initial.s, &self.initial.q, &current_state.a), self.label(&current_state.state.s, &current_state.state.q, &current_state.a));
        while !finished {
            new_state = self.traversal(&current_state);
            print!("p((s:{},q{:?}) , a:{:?}, (s':{},q':{:?}))={}: ", &current_state.state.s, &current_state.state.q, &new_state.a, &new_state.data.s, &new_state.data.q, &new_state.p);
            println!("abstract label: {:?} -> {:?}", self.label(&current_state.state.s, &current_state.state.q, &current_state.a), self.label(&new_state.data.s, &new_state.data.q, &new_state.a));
            current_state = TraversalStateSpace{state: ProductStateSpace{s: new_state.data.s, q: new_state.data.q.to_vec(), mdp_init: false }, a: new_state.a, abstract_label: new_state.abstract_ap.clone(), stoppable: new_state.stoppable };
            // when the task has been completed we need to move onto the next task in the permutation
            if current_state.stoppable {
                // if all of the tasks are finished, then finished becomes true, otherwise
                if tasks_finished(&current_state.abstract_label) {
                    finished = true;
                } // choose the next action to go to
                else {
                    transition_choice = self.determine_choices(&current_state.state.s, &current_state.state.q);
                    let new_choice_state = TraversalStateSpace {
                        state: ProductStateSpace { s: transition_choice.state.s, q: transition_choice.state.q.to_vec(), mdp_init: false },
                        a: TaskAction { a: transition_choice.a.a, task: transition_choice.a.task },
                        abstract_label: transition_choice.abstract_label.clone(),
                        stoppable: transition_choice.stoppable
                    };
                    print!("p((s:{},q{:?}) , a:{:?}, (s':{},q':{:?}))", &current_state.state.s, &current_state.state.q, &new_choice_state.a, &new_choice_state.state.s, &new_choice_state.state.q);
                    println!("abstract label: {:?} -> {:?}", self.label(&current_state.state.s, &current_state.state.q, &current_state.a), self.label(&new_choice_state.state.s, &new_choice_state.state.q, &new_choice_state.a));
                    current_state = new_choice_state
                }
            }
        }
    }

    fn traversal(&self, input: &TraversalStateSpace) -> Traversal {
        // this function is supposed to find paths through a graph
        // starting at the initial state we find a path through the product MDP
        //let a_current: i8 = 1;
        let mut output: Vec<Traversal> = Vec::new();
        for x in self.transitions.iter().filter(|x| x.s == input.state.s && x.q == input.state.q && x.a.a == input.a.a && x.a.task == input.a.task) {
            //println!("State found: ({},{:?},action:{:?}, (s',q'): {:?}, stoppable: {}", x.s, x.q, x.a, x.s_prime, x.stoppable);
            //println!("-> s': {:?}", x.s_prime);
            let o = x.s_prime.choose(&mut rand::thread_rng());
            match o {
                Some(traversal) => output.push(
                    Traversal {
                        a: x.a.clone(),
                        data: ProductStateSpace{
                            s: traversal.s,
                            q: traversal.q.to_vec(),
                            mdp_init: false
                        },
                        p: traversal.p,
                        abstract_ap: traversal.abstract_label.clone(),
                        stoppable: traversal.stoppable
                    }
                ),
                None => println!("nothing")
            }
        }
        let result = output.choose(&mut rand::thread_rng());
        match result {
            Some(x) => Traversal{a: x.a, data: ProductStateSpace{ s: x.data.s, q: x.data.q.to_vec(), mdp_init: false }, p: x.p, abstract_ap: x.abstract_ap.clone(), stoppable: x.stoppable },
            None => {println!("filter was 0 length");Traversal::default()}
        }
    }

    fn label(&self, s: &u32, q: &Vec<u32>, a: &TaskAction) -> Vec<TaskProgress> {
        let mut ap_return: Vec<TaskProgress> = Vec::new();
        for transition in self.transitions.iter().filter(|x| x.s == *s && x.q == *q && x.a.a == a.a && x.a.task == a.task){
            //println!("transtition: {:?}", transition);
            ap_return.extend(transition.abstract_label.values().cloned().collect::<Vec<TaskProgress>>());
        }
        //let ap_return_hash: HashSet<_> = ap_return.iter().cloned().collect();
        //let ap_return_unique: Vec<String> = ap_return_hash.into_iter().collect();
        //ap_return_unique
        ap_return
    }

    pub fn check_transition_index(&self) -> bool {
        for transition in self.transitions.iter() {
            let state: &ProductStateSpace = &self.states[transition.state_index];
            if transition.s != state.s || transition.q != state.q {
                println!("state: {:?}, transition: ({}, {:?})", state, transition.s, transition.q);
                return false
            }
        }
        true
    }
}

#[derive(Debug)]
struct Traversal {
    a: TaskAction, // some action
    data: ProductStateSpace, // some data which is a modified transition
    p: f64,
    abstract_ap: HashMap<u8, TaskProgress>,
    stoppable: bool,
}

#[derive(Debug)]
pub struct TraversalStateSpace {
    state: ProductStateSpace,
    a: TaskAction,
    abstract_label: HashMap<u8, TaskProgress>,
    stoppable: bool
}

impl Traversal {
    fn default () -> Traversal {
        Traversal{
            a: TaskAction{a: -1, task: 0},
            data: ProductStateSpace {
                s: 0,
                q: Vec::new(),
                mdp_init: false,
            },
            p: 0.,
            abstract_ap: HashMap::new(),
            stoppable: false
        }
    }
}

pub struct ProductMDP <'a> {
    pub states: Vec<ProductStateSpace>,
    pub transitions: Vec<ProductTransition>,
    pub initial: ProductStateSpace,
    pub labelling: Vec<fn(u32) -> &'static str>, // TODO determine if the labelling is necessary in the team MDP
    pub task_counter: u8,
    pub dfa_delta: &'a mut HashMap<u8, &'a fn(u32, &'a str) -> u32>, // TODO determine if the dfa delta is necessary in the team MDP
    pub mdp_transitions: &'a Vec<Transition>, // TODO determine if the base mdp transitions are necessary in the team MDP
}

#[derive(Debug, Clone)]
pub struct ModProductTransition{
    pub s: u32,
    pub q: Vec<u32>,
    pub a: TaskAction,
    pub constraint: f64,
    pub reach_rewards: HashMap<u8, f64>,
    pub s_prime: Vec<ModProductTransitionPair>,
    pub ap: Vec<String>,
    pub abstract_label: HashMap<u8, TaskProgress>,
    pub stoppable: bool,
    pub state_index: usize,
}

/*
pub fn generate_task_permuatations(no_tasks: u8) -> Vec<Vec<u8>>{
    let mut task_perms: Vec<Vec<u8>> = Vec::new();
    let mut tasks_ordered: Vec<u8> = (0..no_tasks).collect();
    let task_heap = permutohedron::Heap::new(&mut tasks_ordered);
    for task in task_heap {
        task_perms.push(task.clone())
    }
    task_perms
}*/

impl <'a> ProductMDP <'a> {

    fn determine_choices(&self, s: &u32, q: &Vec<u32>) -> TraversalStateSpace {
        let mut choices: Vec<ProductTransition> = Vec::new();
        for transition in self.transitions.iter().filter(|x| x.s == *s && x.q == *q) {
            choices.push(transition.clone());
        }
        match choices.choose(&mut rand::thread_rng()) {
            Some(x) => TraversalStateSpace {
                state: ProductStateSpace { s: x.s, q: x.q.to_vec(), mdp_init: false },
                a: TaskAction { a: x.a.a, task: x.a.task },
                abstract_label: x.abstract_label.clone(),
                stoppable: x.stoppable
            },
            None => TraversalStateSpace {
                state: ProductStateSpace { s: 0, q: vec![], mdp_init: false },
                a: TaskAction { a: 0, task: 0 },
                abstract_label: Default::default(),
                stoppable: false
            }
        }
    }

    pub fn traverse_n_steps(&self){
        //let mut step: u32 = 0;
        let mut new_state: Traversal; // = Traversal::default();
        let mut finished: bool = false;
        let mut transition_choice = self.determine_choices(&self.initial.s, &self.initial.q);
        println!("transition choice: {:?}", transition_choice);
        let mut current_state = TraversalStateSpace {
            state: ProductStateSpace { s: transition_choice.state.s, q: transition_choice.state.q.to_vec(), mdp_init: false },
            a: TaskAction { a: transition_choice.a.a, task: transition_choice.a.task },
            abstract_label: transition_choice.abstract_label.clone(),
            stoppable: transition_choice.stoppable
        };
        print!("p((s:{},q{:?}) , a:{:?}, (s':{},q':{:?})): ", &self.initial.s, &self.initial.q, &current_state.a, &current_state.state.s, &current_state.state.q);
        println!("abstract label: initial -> {:?}", current_state.abstract_label);
        while !finished {
            new_state = self.traversal(&current_state);
            print!("p((s:{},q{:?}) , a:{:?}, (s':{},q':{:?}))={}: ", &current_state.state.s, &current_state.state.q, &new_state.a, &new_state.data.s, &new_state.data.q, &new_state.p);
            println!("abstract label: {:?} -> {:?}", self.label(&current_state.state.s, &current_state.state.q, &current_state.a), self.label(&new_state.data.s, &new_state.data.q, &new_state.a));
            current_state = TraversalStateSpace{state: ProductStateSpace{s: new_state.data.s, q: new_state.data.q.to_vec(), mdp_init: false }, a: new_state.a, abstract_label: new_state.abstract_ap.clone(), stoppable: new_state.stoppable };
            // when the task has been completed we need to move onto the next task in the permutation
            if current_state.stoppable {
                // if all of the tasks are finished, then finished becomes true, otherwise
                if tasks_finished(&current_state.abstract_label){
                    finished = true;
                } // choose the next action to go to
                else {
                    transition_choice = self.determine_choices(&current_state.state.s, &current_state.state.q);
                    let new_choice_state = TraversalStateSpace {
                        state: ProductStateSpace { s: transition_choice.state.s, q: transition_choice.state.q.to_vec(), mdp_init: false },
                        a: TaskAction { a: transition_choice.a.a, task: transition_choice.a.task },
                        abstract_label: transition_choice.abstract_label.clone(),
                        stoppable: transition_choice.stoppable
                    };
                    print!("p((s:{},q{:?}) , a:{:?}, (s':{},q':{:?}))", &current_state.state.s, &current_state.state.q, &new_choice_state.a, &new_choice_state.state.s, &new_choice_state.state.q);
                    println!("abstract label: {:?} -> {:?}", self.label(&current_state.state.s, &current_state.state.q, &current_state.a), self.label(&new_choice_state.state.s, &new_choice_state.state.q, &new_choice_state.a));
                    current_state = new_choice_state
                }
            }
        }
    }

    fn traversal(&self, input: &TraversalStateSpace) -> Traversal {
        // this function is supposed to find paths through a graph
        // starting at the initial state we find a path through the product MDP
        //let a_current: i8 = 1;
        let mut output: Vec<Traversal> = Vec::new();
        for x in self.transitions.iter().filter(|x| x.s == input.state.s && x.q == input.state.q && x.a.a == input.a.a && x.a.task == input.a.task) {
            //println!("State found: ({},{:?},action:{:?}, (s',q'): {:?}, stoppable: {}", x.s, x.q, x.a, x.s_prime, x.stoppable);
            //println!("-> s': {:?}", x.s_prime);
            let o = x.s_prime.choose(&mut rand::thread_rng());
            match o {
                Some(traversal) => output.push(
                    Traversal {
                        a: x.a.clone(),
                        data: ProductStateSpace{
                            s: traversal.s,
                            q: traversal.q.to_vec(),
                            mdp_init: false
                        },
                        p: traversal.p,
                        abstract_ap: traversal.abstract_label.clone(),
                        stoppable: traversal.stoppable
                    }
                ),
                None => println!("nothing")
            }
        }
        let result = output.choose(&mut rand::thread_rng());
        match result {
            Some(x) => Traversal{a: x.a, data: ProductStateSpace{ s: x.data.s, q: x.data.q.to_vec(), mdp_init: false}, p: x.p, abstract_ap: x.abstract_ap.clone(), stoppable: x.stoppable},
            None => {println!("filter was 0 length");Traversal::default()}
        }
    }

    fn label(&self, s: &u32, q: &Vec<u32>, a: &TaskAction) -> Vec<TaskProgress> {
        let mut ap_return: Vec<TaskProgress> = Vec::new();
        for transition in self.transitions.iter().filter(|x| x.s == *s && x.q == *q && x.a.a == a.a && x.a.task == a.task){
            ap_return.extend(transition.abstract_label.values().cloned().collect::<Vec<TaskProgress>>());
        }
        //let ap_return_hash: HashSet<_> = ap_return.iter().cloned().collect();
        //let ap_return_unique: Vec<String> = ap_return_hash.into_iter().collect();
        //ap_return_unique
        ap_return
    }

    pub fn local_product (&mut self, dfa: &'a DFA<'a>, task: &u8, new_labelling: &'a fn(u32) -> &'a str) -> &'a ProductMDP {
        //the container is a local product MDP in memory, that is we do not need to copy anything
        let mut state_space_new: Vec<ProductStateSpace> = Vec::new();
        let mut transitions_new: Vec<ProductTransition> = Vec::new();
        for state in self.states.iter() {
            for q in dfa.states.iter() {
                let mut newq = state.q.to_vec();
                newq.push(*q);
                state_space_new.push(ProductStateSpace{ s: state.s, q: newq.to_vec(), mdp_init: state.mdp_init });
                for transition in self.transitions.iter().filter(|x| x.s == state.s && x.q == state.q) {
                    //let initial = if transition.s == self.initial.s && transition.q == self.initial.q && *q == dfa.initial { true } else { false };
                    let mut sprimes: Vec<ProductTransitionPair> = Vec::new();
                    let mut sprimes_new: Vec<ProductTransitionPair> = Vec::new();
                    for sprime in transition.s_prime.iter() {
                        // what is the word
                        // the original task is in the transition, and the original function will be used depending on which task is active
                        let qj: usize = usize::try_from(transition.a.task).unwrap();
                        //let state_q_clone: u32 = state.q[qj].clone();
                        //let qprime: u32 = (self.dfa_delta.get(&transition.a.task).unwrap())(state_q_clone, (self.labelling)(sprime.s));
                        let qprime: u32 = (dfa.delta)(*q, (self.labelling[qj])(sprime.s));
                        let mut qprimes: Vec<u32> = sprime.q.to_vec();
                        qprimes.push(qprime);

                        let mut task_progress: HashMap<u8, TaskProgress> = sprime.abstract_label.clone();
                        let task_label = MDP::get_abstract_label(Some(q), &qprime, dfa);
                        task_progress.insert(*task, task_label);

                        sprimes.push(ProductTransitionPair{
                            s: sprime.s,
                            p: sprime.p,
                            q: qprimes,
                            abstract_label: task_progress.clone(),
                            ap: sprime.ap.to_string(),
                            stoppable: is_stoppable(&task_progress)
                        });

                        // new transition
                        let qprime2: u32 = (dfa.delta)(*q, (new_labelling)(sprime.s));
                        let mut qprimes2: Vec<u32> = transition.q.to_vec();
                        qprimes2.push(qprime2);

                        let mut task_progress2: HashMap<u8, TaskProgress> = transition.abstract_label.clone();
                        let task_label2= MDP::get_abstract_label(Some(q), &qprime2, dfa);
                        task_progress2.insert(*task, task_label2);

                        sprimes_new.push(ProductTransitionPair{
                            s: sprime.s,
                            p: sprime.p,
                            q: qprimes2,
                            abstract_label: task_progress2.clone(),
                            ap: format!("{}", (new_labelling)(sprime.s)),
                            stoppable: is_stoppable(&task_progress2)
                        });
                    }
                    let mut state_label: HashMap<u8, TaskProgress> = transition.abstract_label.clone();
                    let state_label_value = MDP::get_abstract_label(None, q, dfa);
                    state_label.insert(*task, state_label_value);
                    transitions_new.push(ProductTransition{
                        s: state.s,
                        q: newq.to_vec() ,
                        a: TaskAction { a: transition.a.a, task: transition.a.task },
                        rewards: transition.rewards,
                        s_prime: sprimes,
                        abstract_label: state_label.clone(),
                        stoppable: is_stoppable(&state_label),
                        ap: transition.ap.to_string(),
                    });
                    transitions_new.push(ProductTransition{
                        s: state.s,
                        q: newq.to_vec() ,
                        a: TaskAction { a: transition.a.a, task: *task },
                        rewards: transition.rewards,
                        s_prime: sprimes_new,
                        abstract_label: state_label.clone(),
                        stoppable: is_stoppable(&state_label),
                        ap: format!("{}", (new_labelling)(state.s)),
                    })
                }
            }
        }
        let mut initial_dfa_vec: Vec<u32> = self.initial.q.to_vec();
        initial_dfa_vec.push(dfa.initial);
        self.states = state_space_new;
        // There will never be an active task in the initial state
        self.initial = ProductStateSpace{s: self.initial.s, q: initial_dfa_vec, mdp_init: true};
        self.transitions = transitions_new;
        self.task_counter = self.task_counter + 1;
        self
    }
}


//###################################################################################
//                                   TEAM MDP SECTION
//###################################################################################
pub struct TeamDFSResult {
    pub visted: Vec<TeamStateSpace>,
    pub not_visited: Vec<TeamStateSpace>
}

struct StateActionPair {
    state: TeamStateSpace,
    action_set: Vec<TaskAction>
}

#[derive(Debug)]
pub struct Transition {
    pub s: u32,
    pub a: i8,
    pub s_prime: Vec<TransitionPair>,
    pub rewards: f64
}

#[derive(Debug)]
pub struct TransitionPair {
    pub s: u32, // need to type it this was so we can automatically reference arrays with states
    pub p: f64
}

#[derive(Debug, Clone, Hash, Eq, PartialEq,)]
pub struct TeamStateSpace {
    pub r: u8,
    pub s: u32,
    pub q: Vec<u32>,
    pub switch_to: bool,
    pub stoppable: bool,
    pub action_set: Vec<TaskAction>,
    pub mdp_init: bool
}

impl fmt::Display for TeamStateSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}, {:?}", self.r, self.s, self.q)
    }
}

#[derive(Debug, Clone)]
pub struct TeamTransitionPair{
    pub r: u8,
    pub s: u32,
    pub p: f64,
    pub q: Vec<u32>,
    pub abstract_label: HashMap<u8, TaskProgress>,
    pub stoppable: bool,
    pub state_index: usize
}

#[derive(Debug, Clone)]
pub struct TeamTransition{
    pub r: u8,
    pub s: u32,
    pub q: Vec<u32>,
    pub a: TaskAction,
    pub rewards_model: Vec<f64>,
    pub s_prime: Vec<TeamTransitionPair>,
    pub abstract_label: HashMap<u8, TaskProgress>,
    pub stoppable: bool,
    pub state_index: usize,
}

impl TeamTransition {
    fn default() -> TeamTransition {
        TeamTransition {
            r: 0,
            s: 0,
            q: vec![],
            a: TaskAction { a: 0, task: 0 },
            rewards_model: Vec::new(),
            s_prime: vec![],
            abstract_label: Default::default(),
            stoppable: false,
            state_index: 0,
        }
    }
}

#[derive(Debug)]
pub struct TeamTraversal {
    pub a: TaskAction,
    pub data: TeamStateSpace,
    pub p: f64,
    pub abstract_label: HashMap<u8, TaskProgress>,
    stoppable: bool
}

#[derive(Debug)]
pub struct TeamTraversalStateSpace {
    state: TeamStateSpace,
    a: TaskAction,
    abstract_label: HashMap<u8, TaskProgress>,
    stoppable: bool,
}

impl TeamTraversalStateSpace {
    fn default() -> TeamTraversalStateSpace {
        TeamTraversalStateSpace {
            state: TeamStateSpace {
                r: 0,
                s: 0,
                q: vec![],
                switch_to: false,
                stoppable: false,
                action_set: Vec::new(),
                mdp_init: false
            },
            a: TaskAction { a: 0, task: 0 },
            abstract_label: HashMap::default(),
            stoppable: false,
        }
    }
}

impl TeamTraversal {
    pub fn default () -> TeamTraversal {
        TeamTraversal {
            a: TaskAction{ a: -1, task: 0 },
            data: TeamStateSpace {
                r: 0,
                s: 0,
                q: Vec::new(),
                switch_to: false,
                stoppable: false,
                action_set: Vec::new(),
                mdp_init: false
            },
            p: 0.0,
            abstract_label: HashMap::new(),
            stoppable: false,
        }
    }
}

pub struct TeamMDP {
    pub states: Vec<TeamStateSpace>,
    pub initial: TeamStateSpace,
    pub transitions: Vec<TeamTransition>,
    pub robot_count: u8,
    pub task_count: u8
}

/// Normal of two vectors
pub fn norm(u: &Vec<f64>, v: &Vec<f64>) -> f64 {
    assert_eq!(u.len(), v.len());
    let mut sum_value: f64 = 0.;
    for (i,_x) in u.iter().enumerate() {
        sum_value += u[i] * v[i]
    }
    sum_value
}

/// The implementation of the team MDP
impl TeamMDP {
    /// Generates a blank Team MDP struct, useful for initially generating the team MDP structure,
    /// and then adding Modified MDP structures to it
    /// # Example
    /// ```
    /// // Generates a new empty team MDP
    /// TeamMDP::empty()
    /// ```
    ///
    pub fn empty() -> TeamMDP {
        TeamMDP {
            states: Vec::new(),
            initial: TeamStateSpace { r: 0, s: 0, q: Vec::new(), switch_to: false, stoppable: false, action_set: Vec::new(), mdp_init: false},
            transitions: Vec::new(),
            robot_count: 0,
            task_count: 0,
        }
    }

    /// Introducing a modified MDP structure to a team MDP structure is the main construction of this file. Adding
    /// a team MDP increases the agent, modifies the state space to identify unique robot states, and adds switch
    /// transitions with the appropriate reward structures
    /// # Example
    /// ```
    /// // First generate a blank Team MDP
    /// use model_checking::TeamMDP;
    /// let mut team: TeamMDP = TeamMDP::empty();
    /// // Suppose that we have a local product local_product1: ModifiedProductMDP
    /// // We introduce this local product specifying the number of agents we want
    /// //to include in the team
    /// team.introduce_modified_mdp(&local_product1, &2); // for a two agent team
    /// ```
    /// We need to include the number of agents from the outside to size the rewards vectors in the team MDP
    /// structure.
    pub fn introduce_modified_mdp(&mut self, mlp: &ModifiedProductMDP, agent_capacity: &u8) -> & mut TeamMDP {
        // create a new robot number
        self.robot_count = &self.robot_count + 1;
        self.task_count = mlp.task_counter;
        let pre_add_statespace: usize = self.states.len();
        let team_states_new = TeamMDP::extend_to_team_product_state_space(&mlp, &self.robot_count);
        self.states.extend(team_states_new);
        //println!("Team state space new: {:?}", self.states);
        // We will always be required to add the transitions of the modified local product MDP so we do that next
        // we cannot just copy the transitions because this will lead to non-identifiable local products

        for transition in mlp.transitions.iter() {
            let mut sprimes: Vec<TeamTransitionPair> = Vec::new();
            let mut rewards_model_values: Vec<f64> = vec![0.; *agent_capacity as usize];
            let mut task_reward_values: Vec<f64> = vec![0.; self.task_count as usize];
            let tasks_remaining = transition.abstract_label.iter().any(|(_k,v)| match v { TaskProgress::Initial => true, TaskProgress::InProgress => true, _ => false});
            if tasks_remaining {
                rewards_model_values[(self.robot_count-1) as usize] = transition.constraint;
            }

            for (k, v) in transition.reach_rewards.iter() {
                task_reward_values[*k as usize] = *v;
            }
            rewards_model_values.append(&mut task_reward_values);

            for sprime in transition.s_prime.iter(){
                sprimes.push(TeamTransitionPair{
                    r: self.robot_count,
                    s: sprime.s,
                    p: sprime.p,
                    q: sprime.q.to_vec(),
                    abstract_label: sprime.abstract_label.clone(),
                    stoppable: sprime.stoppable,
                    state_index: sprime.state_index
                });
            }
            // Probably here we will need to create the switch transitions to another,
            // But! a nuance is that the switch transitions will be from the previous robot to
            // states which conserve task progress in the current robot
            // robot in the team
            // if the robot count is greater than 1 then we can start to add switch transitions
            self.transitions.push(TeamTransition{
                r: self.robot_count,
                s: transition.s,
                q: transition.q.to_vec(),
                a: transition.a,
                rewards_model: rewards_model_values,
                s_prime: sprimes,
                abstract_label: transition.abstract_label.clone(),
                stoppable: transition.stoppable,
                state_index: transition.state_index + pre_add_statespace,
            })
        }

        if self.robot_count > 1 {
            // Mechanically, to add switch transitions to the team MDP model, we add transitions to the team
            // transition vector which satisfy the properties of the switch transition in our definition of the
            // team MDP

            // traversing through the MDP then becomes an exercise of mimicking the scheduler. In any stoppable state
            // I say (the scheduler) show me the actions which I might take at this juncture. This might be to
            //   (i) - Continue the dead task loop
            //  (ii) - Move onto a new task
            // (iii) - Take a switch transition to the new automaton in the team

            // 1. Loop through all of the stoppable states in the previous robot
            let mut switch_transitions: Vec<TeamTransition> = Vec::new();
            let mut state_space_changes: Vec<TeamStateSpace> = Vec::new();
            for prev_r_transition in self.transitions.iter().filter(|x| x.r == self.robot_count - 1 && x.stoppable == true) {
                // 2. We just add a new transition which says that we are now moving to the initial state of the next automaton
                // So I guess this is what I was talking about, we have to add a switch transition to any of the transitions included in our transition vector
                // which have the properties that we are looking for, namely that r_{i-1} + 1 = r, s = 0, qbar = q'bar
                let switch_init_index = self.states.iter().position(|x| x.r == self.robot_count && x.s == 0 && x.q == prev_r_transition.q.to_vec()).unwrap();
                let mut switch_rewards: Vec<f64> = vec![0.; (*agent_capacity + self.task_count) as usize];
                for (task, task_progress) in prev_r_transition.abstract_label.iter() {
                    match task_progress {
                        TaskProgress::JustFailed => { switch_rewards[(*task + agent_capacity) as usize] = 1.},
                        _ => {}
                    }
                }
                let switch_prime = TeamTransitionPair{
                    r: self.robot_count,
                    s: 0,
                    p: 1.,
                    q: prev_r_transition.q.to_vec(),
                    abstract_label: prev_r_transition.abstract_label.clone(),
                    stoppable: prev_r_transition.stoppable,
                    state_index: switch_init_index
                };
                state_space_changes.push(TeamStateSpace{
                    r: self.robot_count,
                    s: 0,
                    q: prev_r_transition.q.to_vec(),
                    switch_to: true,
                    stoppable: prev_r_transition.stoppable,
                    action_set: Vec::new(),
                    mdp_init: false
                });
                switch_transitions.push(TeamTransition{
                    r: self.robot_count - 1,
                    s: prev_r_transition.s,
                    q: prev_r_transition.q.to_vec(),
                    rewards_model: switch_rewards,
                    a: TaskAction { a: 99, task: prev_r_transition.a.task },
                    s_prime: vec![switch_prime],
                    abstract_label: prev_r_transition.abstract_label.clone(),
                    stoppable: prev_r_transition.stoppable,
                    state_index: prev_r_transition.state_index,
                });
            }
            // We need this next bit because we need to be able to easily identify which states can be transitioned to
            for state in state_space_changes.iter() {
                let position: usize = self.states.iter().position(|x| x.s == state.s && x.q == state.q && x.r == state.r).unwrap();
                self.states[position].switch_to = true
            }
            self.transitions.append(&mut switch_transitions);
        } else {
            self.initial.r = self.robot_count;
            self.initial.s = mlp.initial.s;
            self.initial.q = mlp.initial.q.to_vec();
            self.initial.stoppable = true;
            self.initial.action_set = Vec::new();
            self.initial.mdp_init = true
        }
        self
    }

    /// This function is a testing function to demonstrate that each transition is grounded in the correct state.
    /// This is done through the use of the ```TeamTransition::state_index``` attribute, which represents the location of the state in the
    /// team state space vector.
    pub fn check_transition_index(&self) -> bool {
        for transition in self.transitions.iter() {
            let state: &TeamStateSpace = &self.states[transition.state_index];
            if state.s != transition.s || state.q != transition.q {
                println!("error: state: {:?} -> transition: ({},{},{:?})", state, transition.r, transition.s, transition.q);
                return false
            }
        }
        true
    }

    /// This is a helper function for the modified product MDP before it enters the team
    pub fn extend_to_team_product_state_space(local_prod: &ModifiedProductMDP, rbot_number: &u8) -> Vec<TeamStateSpace> {
        let mut team_state_space: Vec<TeamStateSpace> = Vec::new();
        let rnew: &u8 = rbot_number;
        for state in local_prod.states.iter() {
            let next_transition = local_prod.transitions.iter().filter(|x| x.s == state.s && x.q == state.q ).next();
            match next_transition {
                Some(x) => {
                    team_state_space.push(TeamStateSpace{
                        r: *rnew,
                        s: state.s,
                        q: state.q.to_vec(),
                        switch_to: false,
                        stoppable: x.stoppable,
                        action_set: Vec::new(),
                        mdp_init: state.mdp_init
                    });
                },
                None => {
                    println!("Received error on searching for a transition for the state {},{:?} in the Mod Product MDP", state.s, state.q);
                    team_state_space.push(TeamStateSpace{
                        r: *rnew,
                        s: state.s,
                        q: state.q.to_vec(),
                        switch_to: false,
                        stoppable: false,
                        action_set: Vec::new(),
                        mdp_init: state.mdp_init
                    });
                }
            }
        }
        team_state_space
    }

    /// This is a private inner function of the Expected Total Rewards Optimisation, this is the first
    /// part of the algorithm (2) which optimises the scheduler for a set of task and constraint rewards
    fn inner_action_optimisation(&self, state: &TeamStateSpace, xbar: &Vec<f64>, w_arr1: &Array1<f64>, reachable_states: &Vec<TeamStateSpace>, verbose: &bool) -> (f64, TaskAction) {
        let mut action_values: Vec<f64> = Vec::new();
        let mut actions: Vec<TaskAction> = Vec::new();
        if *verbose {
            println!("state: ({},{},{:?})", state.r, state.s, state.q);
        }
        for action in state.action_set.iter(){
            for transition in self.transitions.iter().
                filter(|x| x.r == state.r && x.s == state.s && x.q == state.q && x.a.a == action.a && x.a.task == action.task) {
                let rewards = arr1(&transition.rewards_model);
                let norm = rewards.dot(w_arr1);
                let mut sum_sprime_values: Vec::<f64> = Vec::new();
                for sprime in transition.s_prime.iter(){
                    let sprime_position = reachable_states.iter().position(|x| x.r == sprime.r && x.s == sprime.s && x.q == sprime.q).unwrap();
                    sum_sprime_values.push(sprime.p * xbar[sprime_position]);
                    if *verbose {
                        println!("sprime: ({},{},{:?}), p: {}, xbar: {}:", sprime.r, sprime.s, sprime.q, sprime.p, xbar[sprime_position]);
                    }
                }
                let summed_values: f64 = sum_sprime_values.iter().sum();
                //println!("summed value: {}", summed_values);
                action_values.push(norm + summed_values);
                actions.push(*action);
            }
        }
        if *verbose {
            println!("");
        }
        //println!("Action values: {:?}", action_values);
        //println!("Actions: {:?}", actions);
        let non_nan_action_values: Vec<_> = action_values.iter().cloned().map(NotNan::new).filter_map(Result::ok).collect();
        if non_nan_action_values.is_empty() {
            println!("state with empty actions: ({},{},{:?}), actions: {:?}",  state.r, state.s, state.q, state.action_set);
        }
        let min_value = non_nan_action_values.iter().min().unwrap();
        let index = non_nan_action_values.iter().position(|i| i == min_value).unwrap();
        let ynew: f64 = min_value.into_inner();
        let opt_action: TaskAction = actions[index];
        (ynew, opt_action)
    }

    /// This private function is the second loop calculation of algorithm (2) which calculates the
    /// actual reward for each objective in the multi-objective problem for a given scheduler
    fn inner_optimal_reward_optimisation(&self, mu: &mut Vec<(TeamStateSpace, TaskAction)>, state: &TeamStateSpace, X: &mut Vec<Vec<f64>>, Y: &mut Vec<Vec<f64>>, w: &Vec<f64>, reachable_states: &Vec<TeamStateSpace>, k: &usize) -> () {
        let optimal_action = mu[*k].1;
        for transition in self.transitions.iter().
            filter(|x| x.r == state.r && x.s == state.s && x.q == state.q && x.a.a == optimal_action.a && x.a.task == optimal_action.task) {
            for j in 0..w.len() {
                let mut sum_values: Vec<f64> = Vec::new();
                for sprime in transition.s_prime.iter() {
                    let sprime_position: usize = reachable_states.iter().position(|x| x.r == sprime.r && x.s == sprime.s && x.q == sprime.q).unwrap();
                    sum_values.push(sprime.p * X[j][sprime_position])
                }
                let sum_value: f64 = sum_values.iter().sum();
                Y[j][*k] = transition.rewards_model[j] + sum_value;
            }
        }
    }

    /// This is algorithm 2, of the MOTAP theory, see paper for more details, and formalisations
    #[inline]
    pub fn minimise_expected_weighted_cost_of_scheduler(&self, reachable_states: &Vec<TeamStateSpace>, w: &Vec<f64>, epsilon_0: f64) -> (Vec<(TeamStateSpace, TaskAction)>, Vec<f64>) {
        // initialise the action vector
        let mut mu: Vec<(TeamStateSpace, TaskAction)> = vec![(TeamStateSpace{
            r: 0,
            s: 0,
            q: vec![],
            switch_to: false,
            stoppable: false,
            action_set: Vec::new(),
            mdp_init: false
        }, TaskAction::default()); reachable_states.len()];
        let mut r: Vec<f64> = vec![0.; w.len()];
        let w_arr1 = arr1(w);
        let m = self.robot_count;
        //println!("robot capacity: {}", m);
        let mut epsilon: f64 = 1.;
        //for i in (1..m).rev() {
            // initialise a bunch of vectors
        for i in (1..2+1).rev() {
            //let i: u8 = 1;
            //println!("generating the scheduler for robot: {}", i);
            let mut xbar: Vec<f64> = vec![0.; reachable_states.len()];
            let mut X: Vec<Vec<f64>> = vec![vec![0.; reachable_states.len()]; w.len()];
            let mut ybar: Vec<f64> = vec![0.; reachable_states.len()];
            let mut Y: Vec<Vec<f64>> = vec![vec![0.; reachable_states.len()]; w.len()];
            while epsilon > epsilon_0 {
                /*if i < m {
                    for (k, state) in reachable_states.iter().enumerate().filter(|(ii,x)| x.r == i || (x.r == i + 1 && x.switch_to)){
                        //let state_position: usize = reachable_states.iter().position(|x| x.r == state.r && x.s == state.s && x.q == state.q).unwrap();
                        //assert_eq!(state_position, k);
                        let (ynew, opt_action) = self.inner_action_optimisation(state, &mut xbar, &w_arr1, reachable_states, &false);
                        ybar[k] = ynew;
                        mu[k] = (state.clone(), opt_action);
                    }
                } else {

                 */
                for (k,state) in reachable_states.iter().enumerate() {//.filter(|(ii,x)| x.r == i){
                    //let state_position: usize = reachable_states.iter().position(|x| x.r == state.r && x.s == state.s && x.q == state.q).unwrap();
                    //assert_eq!(state_position, k);
                    let (ynew, opt_action) = self.inner_action_optimisation(state, &mut xbar, &w_arr1, reachable_states, &false);
                    ybar[k] = ynew;
                    mu[k] = (state.clone(), opt_action);
                }
                //}
                let xbar_arr1 = arr1(&xbar);
                let ybar_arr1 = arr1(&ybar);
                let diff = &ybar_arr1 - &xbar_arr1;
                let non_nan_eps: Vec<_> = diff.iter().cloned().map(NotNan::new).filter_map(Result::ok).collect();
                let epsilon_new = non_nan_eps.iter().max().unwrap().into_inner();
                //println!("max epsilon: {}", epsilon_new);
                xbar = ybar.to_vec();
                epsilon = epsilon_new
            }
            epsilon = 1.;
            while epsilon > epsilon_0 {
                /*if i < m {
                    for (k,state) in reachable_states.iter().enumerate().filter(|(ii, x)| x.r == i || (x.r == i + 1 && x.switch_to)) {
                        //let state_position = reachable_states.iter().position(|x| x.r == state.r && x.s == state.s && x.q == state.q).unwrap();
                        self.inner_optimal_reward_optimisation(&mut mu, state, &mut X, &mut Y, &w, &reachable_states, &k);
                    }
                } else {

                 */
                for (k,state) in reachable_states.iter().enumerate() {//.filter(|(ii, x)| x.r == i) {
                    //let state_position = reachable_states.iter().position(|x| x.r == state.r && x.s == state.s && x.q == state.q).unwrap();
                    self.inner_optimal_reward_optimisation(&mut mu, state, &mut X, &mut Y, &w, &reachable_states, &k);
                }
                //}
                let mut eps_j_values: Vec<f64> = vec![1.; w.len()];

                for j in 0..w.len() {
                    let ybar_j_arr1 = arr1(&Y[j]);
                    let xbar_j_arr1 = arr1(&X[j]);
                    let diff = &ybar_j_arr1 - &xbar_j_arr1;
                    let non_nan_eps: Vec<_> = diff.iter().cloned().map(NotNan::new).filter_map(Result::ok).collect();
                    let epsilon_j = non_nan_eps.iter().max().unwrap().into_inner();
                    eps_j_values[j] = epsilon_j;
                    X[j] = Y[j].to_vec();
                }
                let non_nan_eps_total: Vec<_> = eps_j_values.iter().cloned().map(NotNan::new).filter_map(Result::ok).collect();
                let epsilon_new = non_nan_eps_total.iter().max().unwrap().into_inner();

                //println!("max epsilon: {}", epsilon_new);
                xbar = ybar.to_vec();
                epsilon = epsilon_new
            }
            epsilon = 1.;
            if i == 1 {
                for j in 0..w.len() {
                    let initial_state: usize = reachable_states.iter().position(|x| x.r == self.initial.r && x.s == self.initial.s && x.q == self.initial.q).unwrap();
                    r[j] = Y[j][initial_state];
                    //print!("y[{}] = {}, ", j, r[j]);
                }
            }
            //println!("");
        }
        // find the position of the initial state
        //reachable_states.iter().position(|x| x.r == i && x.s == self.initial.s && )
        (mu, r)
    }

    fn inner_reachability(&self, queue: &mut VecDeque<TeamStateSpace>, transition: &TeamTransition, visited: &mut Vec<bool>) -> () {
        for sprime in transition.s_prime.iter(){
            // if the mission was complete, record the active task because we will use this to search
            // for actions which led to dead loops
            let sprime_index = self.states.iter().position(|x| x.r == sprime.r && x.s == sprime.s && x.q == sprime.q).unwrap();
            if !visited[sprime_index]{
                visited[sprime_index] = true;

                queue.push_front(TeamStateSpace {
                    r: sprime.r,
                    s: sprime.s,
                    q: sprime.q.to_vec(),
                    switch_to: false,
                    stoppable: false,
                    action_set: Vec::new(),
                    mdp_init: self.states[sprime_index].mdp_init
                });
            }
        }
    }

    /// DFS for finding the reachable states from the initial state in the team vector, ALSO for associating a task with a
    /// state, the definition of our team structure says that there can only be one relevant task per action set for a state.
    /// The initial state is the only state where there is no task active.
    pub fn reachable_states(&mut self) -> TeamDFSResult {
        // We assign a task by looking at the action is took to get to that task and then assign
        // that task to the state. In the case where the task is finished or failed, unless the base
        // MDP is back to its initial state, and then we generate a new task set minus the current task

        let mut queue: VecDeque<TeamStateSpace> = VecDeque::new();
        let mut visited_states: Vec<TeamStateSpace> = Vec::new();
        let mut dead_states: Vec<TeamStateSpace> = Vec::new();

        queue.push_front(self.initial.clone());
        let position_init = self.states.iter().position(|x| x.r == self.initial.r && x.s == self.initial.s && x.q == self.initial.q).unwrap();
        let mut visited: Vec<bool> = vec![false; self.states.len()];
        // record a description of the states visited
        let mut mod_states: Vec<TeamStateSpace> = self.states.to_vec();
        visited[position_init] = true;
        while !queue.is_empty() {
            let next_state = queue.pop_front().unwrap();
            let nextstate_index: usize = mod_states.iter().position(|x| x.r == next_state.r && x.s == next_state.s && x.q == next_state.q).unwrap();
             //println!("looking for transitions to state: ({},{},{:?})", next_state.r, next_state.s, next_state.q);

            // CASES
            // The mdp state is initial
            // 1. There is at least one task remaining
            // 2. There are no tasks remaining
            // Otherwise there is a task in progress

            // how many tasks are available
            let abstract_label = &self.transitions.iter().filter(|x| x.r == next_state.r && x.s == next_state.s && x.q == next_state.q).next().unwrap().abstract_label;
            // which tasks are left to complete
            // check that in progress is not in the state's abstract label
            let key: Option<u8> =  abstract_label.iter().find_map(|(k,v)| match v { TaskProgress::InProgress => Some(*k), _ => None }); // There can only be one, by definition, otherwise it is illegal
            //println!("abstract label: {:?}", abstract_label);
            match key {
                Some(k) => {
                    //println!("State: ({},{},{:?}), active task: {}", next_state.r, next_state.s, next_state.q, k);
                    let mut action_set: HashSet<TaskAction> = HashSet::new();
                    for transition in self.transitions.iter().
                        filter(|x| x.r == next_state.r && x.s == next_state.s && x.q == next_state.q && x.a.task == k) {
                        action_set.insert(transition.a);
                        self.inner_reachability(&mut queue, transition, &mut visited);
                    }
                    mod_states[nextstate_index].action_set = action_set.into_iter().collect();
                },
                None => {
                    // If there are no tasks in progress then there are still two choices to eliminate,
                    // 1. there are no tasks remaining which we currently cannot handle
                    // 2. There are tasks in the initial state then we need to go through all of
                    //    those task actions
                    let remaining_keys: Vec<u8> = abstract_label.iter().filter_map(|(k,v)| match v { TaskProgress::Initial => Some(*k), _ => None}).collect();

                    // todo there is actually another condition here, in that is the abstract label is fail, just fail or complete,
                    //  we are getting closer to the solution here in that, we know that we have to look at the previous action to determinine, which task gets
                    //  continued when we can't gether any information about the previous task
                    //  What are the steps in a robot
                    //  t: InProgress -> ... -> justFail | complete -> ... [here is where we are interested in what to do next] -> init (new task) ->
                    if remaining_keys.is_empty() {
                        // this is still the most puzzling of the cases in that it is unclear how to continue in dead states
                        //println!("State: ({},{},{:?}), there are no tasks remaining, continuing with no task", next_state.r, next_state.s, next_state.q);
                        // search for state actions which led to this state
                        let mut action_set: HashSet<TaskAction> = HashSet::new();
                        for transition in self.transitions.iter().filter(|x| x.r == next_state.r && x.s == next_state.s && x.q == next_state.q) {
                            action_set.insert(transition.a);
                            self.inner_reachability(&mut queue, transition, &mut visited);
                        }
                        mod_states[nextstate_index].action_set = action_set.into_iter().collect();
                    } else {
                        // if the remaining tasks are not empty, but the robot is not initial, then we need to determine what the previous action was
                        //println!("State: ({},{},{:?}), remaining tasks: {:?}", next_state.r, next_state.s, next_state.q, remaining_keys);
                        if next_state.mdp_init {
                            //println!("Re-init: ({},{},{:?}), Remaining tasks: {:?}", next_state.r, next_state.s, next_state.q, remaining_keys);
                            // select a new task to complete
                            let mut action_set: HashSet<TaskAction> = HashSet::new();
                            for task in remaining_keys.iter() {
                                for transition in self.transitions.iter().
                                    filter(|x| x.r == next_state.r && x.s == next_state.s && x.q == next_state.q && x.a.task == *task) {
                                    //println!("transition: {:?}", transition);
                                    action_set.insert(transition.a);
                                    self.inner_reachability(&mut queue, transition, &mut visited);
                                }
                            }
                            // find the position of the queued state in set of states and alter the actions available to this state
                            mod_states[nextstate_index].action_set = action_set.into_iter().collect();
                        } else {
                            // what was the previous action that led to this state

                            let mut tasks: HashSet<u8> = HashSet::new();
                            for prev_transition in self.transitions.iter().filter(|x| x.s_prime.iter().filter(|z| z.r == next_state.r && z.s == next_state.s && z.q == next_state.q).peekable().peek().is_some()){
                                //println!("transition: {:?}", prev_transition);
                                let state_position: usize = self.states.iter().position(|x| x.r == prev_transition.r && x.s == prev_transition.s && x.q == prev_transition.q).unwrap();
                                //println!("state position: {}, visited: {}, action: {:?}", state_position, visited[state_position], prev_transition.a);
                                if visited[state_position] {
                                    let label = abstract_label.get(&prev_transition.a.task).unwrap();
                                    //let label_prev = abs
                                    //println!("label: {:?}, prev label: {:?}, task: {}, {:?}", abstract_label, prev_transition.abstract_label, prev_transition.a.task, label);
                                    let legal: bool = match label {
                                        TaskProgress::Initial => false,
                                        _ => true
                                    };
                                    //println!("legal: {}", legal);
                                    if legal {
                                        //println!(" previous state: ({},{},{:?}), action: {:?}", prev_transition.r, prev_transition.s, prev_transition.q, prev_transition.a);
                                        tasks.insert(prev_transition.a.task);
                                    }
                                }
                            }
                            //println!("tasks: {:?}", tasks);
                            let mut action_set: HashSet<TaskAction> = HashSet::new();
                            for task in tasks.iter() {
                                for transition in self.transitions.iter().
                                    filter(|x| x.r == next_state.r && x.s == next_state.s && x.q == next_state.q && x.a.task == *task) {
                                    action_set.insert(transition.a);
                                    self.inner_reachability(&mut queue, transition, &mut visited);
                                }
                            }
                            //println!("actions: {:?}", action_set);
                            mod_states[nextstate_index].action_set = action_set.into_iter().collect();
                        }
                    }
                }
            }
        }

        for (i,x) in visited.iter().enumerate() {
            if *x {
                visited_states.push(mod_states[i].clone());
            } else {
                dead_states.push(mod_states[i].clone());
            }
        }
        TeamDFSResult {
            visted: visited_states,
            not_visited: dead_states,
        }
    }

    /// Generate a graph on the reachable states of the team MDP, useful for
    /// debugging the team MDP structure
    pub fn generate_graph(&self, visited_states: &mut Vec<TeamStateSpace>) -> String {
        // todo we need one final bit which is the dead loops, the bit where both states are complete
        //  but we continue the loop anyway
        let mut graph: Graph<String, String> = Graph::new();
        let mut node_added: Vec<bool> = vec![false; visited_states.len()];
        //let mut state_action_pairs: Vec<StateActionPair> = Vec::new();
        for state in visited_states.iter() {
            let origin_index = visited_states.iter().position(|x| x.r == state.r && x.s == state.s && x.q == state.q).unwrap();
            if !node_added[origin_index] {
                graph.add_node(format!("({},{},{:?})", state.r, state.s, state.q));
                node_added[origin_index] = true;
            }
            // A new task is to be chosen
            // we have to get the abstract label of the next state to see which tasks
            // have not been finished/failed yet
            for action in state.action_set.iter() {
                for transition in self.transitions.iter().
                    filter(|x| x.r == state.r && x.s == state.s && x.q == state.q && x.a.a == action.a && x.a.task == action.task){
                    for sprime in transition.s_prime.iter() {
                        let destination_index = match visited_states.iter().position(|x| x.r == sprime.r && x.s == sprime.s && x.q == sprime.q){
                            Some(x) => x,
                            None => {panic!("state: ({},{},{:?}) at action: {:?} to s': ({},{},{:?})", state.r, state.s, state.q, action, sprime.r, sprime.s, sprime.q)}
                        };
                        if !node_added[destination_index] {
                            graph.add_node(format!("({},{},{:?})", sprime.r, sprime.s, sprime.q));
                            node_added[destination_index] = true;
                        }
                        let action = format!("a: {}, task: {}", transition.a.a, transition.a.task);
                        let origin_node_index = graph.node_indices().find(|i| graph[*i] == format!("({},{},{:?})", state.r, state.s, state.q)).unwrap();
                        let destination_node_index = graph.node_indices().find(|i| graph[*i] == format!("({},{},{:?})", sprime.r, sprime.s, sprime.q)).unwrap();
                        graph.add_edge(origin_node_index, destination_node_index, action);
                    }
                }
            }
        }
        format!("{}", Dot::new(&graph))
    }

    pub fn construct_scheduler_graph(&self, mu: &Vec<(TeamStateSpace, TaskAction)>) -> String {
        let mut graph: Graph<String, String> = Graph::new();
        let mut node_added: Vec<bool> = vec![false; mu.len()];

        for (k, (state, action)) in mu.iter().enumerate() {
            //println!("State: {}, action: {:?}", format!("({},{},{:?})", state.r, state.s, state.q), action);
            if !node_added[k] && !state.q.is_empty() {
                graph.add_node(format!("({},{},{:?})", state.r, state.s, state.q));
               // println!("State added: {}", format!("({},{},{:?})", state.r, state.s, state.q));
                node_added[k] = true;
            }
            for transition in self.transitions.iter().
                filter(|x| x.r == state.r && x.s == state.s && x.q == state.q && x.a.a == action.a && x.a.task == action.task) {
                for sprime in transition.s_prime.iter() {
                    let sprime_position = match mu.iter().position(|(x,_y)| x.s == sprime.s && x.r == sprime.r && x.q == sprime.q) {
                        Some(x) => x,
                        None => {println!("s': ({},{},{:?}) is not contained in the scheduler", sprime.r, sprime.s, sprime.q); 0}
                    };
                    if !node_added[sprime_position] {
                        graph.add_node(format!("({},{},{:?})", sprime.r, sprime.s, sprime.q));
                        node_added[sprime_position] = true
                    }
                    let action_str = format!("a: {}, task: {}", action.a, action.task);
                    let origin_index_node = graph.node_indices().find(|i| graph[*i] == format!("({},{},{:?})", state.r, state.s, state.q)).unwrap();
                    let destinatin_node_index = graph.node_indices().find(|i| graph[*i] == format!("({},{},{:?})", sprime.r, sprime.s, sprime.q)).unwrap();
                    graph.add_edge(origin_index_node, destinatin_node_index, action_str);
                }
            }
        }
        format!("{}", Dot::new(&graph))
    }
}

//####################################################################################
//                                   GENERAL FUNCTIONS
//####################################################################################

/// Standard value iteration approach to mdp reachability model checking
pub fn value_iteration(states:&Vec<u32>, transitions: &Vec<Transition>, epsilon: f64, target: &Vec<u32>, s0min: &Vec<u32>) -> Result<Vec<f64>, String> {
    let mut delta: f64 = 1.;
    let l = states.len();
    let mut x: Vec<f64> = vec![0.; l];
    let mut xprime: Vec<f64> = vec![0.; l];
    for t in target.iter() {
        let t_index = usize::try_from(*t).unwrap();
        x[t_index] = 1.;
        xprime[t_index] = 1.;
    }
    let mut r: Vec<u32> = vec![];
    for s in states.iter() {
        if !s0min.contains(s) && !target.contains(s) {
            r.push(*s)
        }
    }
    //println!("calculable states: {:?}", r);
    while delta > epsilon {
        for s in r.iter() {
            // The next part of this problem is filtering dynamic arrays of custom structs
            {
                let s_index= usize::try_from(*s).unwrap();
                //println!("s_index: {:?}", s_index);
                let mut choose_arr: Vec<f64> = Vec::new();
                for n in transitions.iter().filter(|x| x.s == *s) {
                    //println!("{:?}", n);
                    // we need the transitions from
                    let mut sumarr: Vec<f64> = Vec::new();
                    for transition in n.s_prime.iter() {
                        let sprime_index = usize::try_from(transition.s).unwrap();
                        sumarr.push(transition.p * x[sprime_index])
                    }
                    //println!("{:?}", sumarr);
                    choose_arr.push(sumarr.iter().sum());
                }
                //println!("sum s,a -> s': {:?}", choose_arr);
                choose_arr.sort_by(|a,b| b.partial_cmp(a).unwrap()); // b before a greater than, otherwise less than ordering
                //println!("sorted double arr: {:?}", choose_arr);
                let max_a = choose_arr[0];
                xprime[s_index] = max_a;
            }
        }
        //println!("x' = {:?}", xprime);
        let mut delta_v: Vec<f64> = vec![0.; x.len()];
        for s in states.iter() {
            let s_index = usize::try_from(*s).unwrap();
            delta_v[s_index] = xprime[s_index] - x[s_index]
        }
        delta_v.sort_by(|a,b| b.partial_cmp(a).unwrap());
        delta = delta_v[0];
        //println!("delta {}", delta);
        for s in r.iter() {
            let s_index = usize::try_from(*s).unwrap();
            x[s_index] = xprime[s_index];
        }
    }
    Ok(x)
}

/// Generate a witness vector such that the pareto face satisfies the target vector
pub fn witness(target: &Vec<f64>, closure_set: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut problem = Problem::new(OptimizationDirection::Maximize);

    let mut vars: HashMap<String, Variable> = HashMap::new();
    for i in 0..closure_set.len() {
        vars.insert(format!("v{}", i), problem.add_var(0., (0., 1.)));
    }
    //println!("vars: {:?}", vars);

    for i in 0..target.len() {
        let mut lhs = LinearExpr::empty();
        for j in 0..closure_set.len(){
            lhs.add(*vars.get(&*format!("v{}",j)).unwrap(), closure_set[j][i]);
        }
        //println!("lhs: {:?}", lhs);
        problem.add_constraint(lhs, ComparisonOp::Le, target[i]);
    }

    let mut lhs = LinearExpr::empty();
    for i in 0..closure_set.len(){
        lhs.add(*vars.get(&*format!("v{}", i)).unwrap(), 1.0);
    }
    problem.add_constraint(lhs, ComparisonOp::Eq, 1.0);
    let solution = problem.solve().unwrap();
    let mut v: Vec<f64> = vec![0.; closure_set[0].len() as usize + 1];
    for i in 0..closure_set.len() {
        let val: f64 = ((solution[*vars.get(&*format!("v{}", i)).unwrap()] * 1000.).round() / 1000.) as f64;
        v[i] = val;
        //println!("index: {}, v: {:?}", i, v);
    }
    v
}

/// This function will return a point on the pareto frontier according to the data input. Essentially
/// this is a linear separability problem, where we want to discriminate +ve, and -ve points.
pub fn pareto_lp(h: &Vec<Vec<f64>>, k: &Vec<Vec<f64>>, dim: &u8) -> Vec<f64> {
    let mut problem = Problem::new(OptimizationDirection::Maximize);

    let mut vars: HashMap<String, Variable> = HashMap::new();
    for i in 0..*dim {
        vars.insert(format!("w{}", i), problem.add_var(0., (0., 1.)));
    }
    vars.insert(format!("delta"), problem.add_var(1.0, (f64::NEG_INFINITY, f64::INFINITY)));
    let b = problem.add_var(0., (f64::NEG_INFINITY, f64::INFINITY));
    for x in h.iter() {
        let mut lhs = LinearExpr::empty();
        for j in 0..*dim {
            lhs.add(*vars.get(&*format!("w{}", j)).unwrap(), x[j as usize]);
        }
        lhs.add(b, 1.0);
        lhs.add(*vars.get("delta").unwrap(), -1.0);
        problem.add_constraint(lhs, ComparisonOp::Ge, 0.);
    }
    for x in k.iter() {
        let mut lhs = LinearExpr::empty();
        for j in 0..*dim {
            lhs.add(*vars.get(&*format!("w{}", j)).unwrap(), x[j as usize]);
        }
        lhs.add(b, 1.0);
        lhs.add(*vars.get("delta").unwrap(), 1.0);
        problem.add_constraint(lhs, ComparisonOp::Le, 0.);
    };
    let mut lhs = LinearExpr::empty();
    for j in 0..*dim {
        lhs.add(*vars.get(&*format!("w{}", j)).unwrap(), 1.0)
    }
    problem.add_constraint(lhs, ComparisonOp::Eq, 1.);

    let solution = problem.solve().unwrap();
    let mut w: Vec<f64> = vec![0.; *dim as usize];
    for i in 0..*dim {
        let w_val: f64 = ((solution[*vars.get(&*format!("w{}", i)).unwrap()] * 1000.).round() / 1000.) as f64;
        w[i as usize] = w_val;
    }
    w
}

pub fn generate_random_vector_sum1(n: &u32, lower_bound: &u64, upper_bound: &u64) -> Vec<f64> {
    let mut rng = rand::thread_rng();

    let vals: Vec<u64> = (0..*n).map(|_| rng.gen_range(*lower_bound,*upper_bound)).collect();
    let sum_vals: u64 = vals.iter().sum();
    let f64_vals: Vec<f64> = vals.into_iter().map(|x| (x as f64) / (sum_vals as f64)).collect();
    f64_vals
}

pub fn member_closure_set(hull_set: &Vec<Vec<f64>>, r: &Vec<f64>) -> bool {
    for x in hull_set.iter() {
        let upward_closure = x.iter().zip(r).all(|(x,z)| x <= z);
        if upward_closure {
            return true
        }
    }
    false
}

pub struct Alg1Output {
    pub hullset: Vec<Vec<f64>>,
    pub mus: Vec<Vec<(TeamStateSpace, TaskAction)>>
}
/// Algorithm 1, and the main algorithm in our collection of algorithms, it takes a team
/// MDP structure (fully formed inclusive of the rewards model), and a target vector to
/// generate a series of pareto optimal schedulers on the vertices of the polytope formed
/// in R^n (n objectives)
#[inline]
pub fn muliobj_scheduler_synthesis(team: &TeamMDP, target: &Vec<f64>, reachable_states: &Vec<TeamStateSpace>) -> Alg1Output {
    let mut hull_set: Vec<Vec<f64>> = Vec::new();
    let mut mus: Vec<Vec<(TeamStateSpace, TaskAction)>> = Vec::new();
    let target_set = vec![target.to_vec()];
    let mut w = vec![0.25, 0.25, 0.25, 0.25];
    for k in 0..10 {
        //let w = generate_random_vector_sum1(&4, &0, &100);
        //println!("w: {:?}", w);
        let (mu, r) = team.minimise_expected_weighted_cost_of_scheduler(reachable_states, &w, 0.001);
        hull_set.push(r);
        mus.push(mu);
        w = pareto_lp(&hull_set, & target_set, &4);
        //println!("{:?}", w);
        //println!("r: {:?}", r);
        //println!("output norm: {}", arr1(&w).dot(&arr1(&r)));
        //println!("target norm: {}", arr1(&w).dot(&arr1(&target)));
        if member_closure_set(&hull_set, &target) {
            //println!("target is a member of the upward closure, found in {} steps", k+1);
            break
        }
    }
    //println!("upward closure: {:?}", hull_set);
    let v: Vec<f64> = witness(&target, &hull_set);
    //println!("v: {:?}", v);
    Alg1Output {
        hullset: hull_set,
        mus: mus
    }
}