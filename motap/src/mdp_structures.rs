use std::collections::{HashSet, HashMap};
use rand::seq::SliceRandom;
use std::convert::TryFrom;
use std::hash::Hash;
use std::fmt;
use crate::team_mdp_structures::TeamMDP;

#[derive(Debug)]
pub struct TransitionPair {
    pub s: u32, // need to type it this was so we can automatically reference arrays with states
    pub p: f32
}

#[derive(Debug)]
pub struct Transition {
    pub s: u32,
    pub a: i8,
    pub s_prime: Vec<TransitionPair>,
    pub rewards: f32
}

#[derive(Debug, Clone)]
pub struct ProductTransitionPair {
    pub s: u32,
    pub p: f32,
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
}

impl ProductStateSpace {
    pub fn append_state(&mut self, state: u32) {
        self.q.push(state);
    }

    pub fn default() -> ProductStateSpace {
        ProductStateSpace {
            s: 0,
            q: Vec::new(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
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
    pub rewards: f32,
    pub s_prime: Vec<ProductTransitionPair>,
    pub abstract_label: HashMap<u8, TaskProgress>,
    // we actually gain nothing by storing which task is a just fail because construction
    // of the modified product vector relies on looking at all s' from s
    pub stoppable: bool,
    //pub accepted: bool,
    //pub rejected: bool,
    pub ap: String,
}

pub struct Pair {
    pub q: u32,
    pub a: Vec<char>
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
    let inProgress = abstract_label.values().any(|x| match x {
        TaskProgress::InProgress => true,
        _ => false,
    });
    if inProgress {
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
        let mut state_index: u8 = 0;
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
                state_space_new.push(ProductStateSpace{ s: *state, q: vec![*q]});
            }
        }
        empty_container.dfa_delta.insert(0u8, &dfa.delta);
        empty_container.states = state_space_new;
        empty_container.transitions = transitions_new;
        empty_container.initial = ProductStateSpace{ s: self.initial, q: vec![dfa.initial]};
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
    pub p: f32,
    pub q: Vec<u32>,
    pub abstract_label: HashMap<u8, TaskProgress>,
    pub ap: String,
    pub state_index: usize,
    pub stoppable: bool,
}

pub fn tasks_finished(transition: &HashMap<u8, TaskProgress>) -> bool {
    let mut task_progress: Vec<bool> = Vec::new();
    for (k, v) in transition {
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

    pub fn task_rewards(abstract_label: &HashMap<u8, TaskProgress>) -> HashMap<u8, f32> {
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
            unique_state_space.insert(ProductStateSpace{ s: transition.s, q: transition.q.to_vec()});
            let loop_cond: bool = transition.s_prime.iter().any(|x| x.s == transition.s);
            let no_loop_cond: bool = transition.s_prime.iter().any(|x| x.s != transition.s);
            if loop_cond == true && no_loop_cond == true {
            //if transition.self_loop == true {
                // add a new state to the state space, this state will be snew
                unique_state_space.insert(ProductStateSpace{ s: counter, q: transition.q.to_vec() });
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
                    abstract_label: transition.abstract_label.clone(),
                    ap: Vec::new(),
                    stoppable: is_stoppable(&transition.abstract_label),
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
                                    unique_state_space.insert(ProductStateSpace{ s: counter, q: sprime.q.to_vec() });
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
                state: ProductStateSpace { s: x.s, q: x.q.to_vec() },
                a: TaskAction { a: x.a.a, task: x.a.task },
                abstract_label: x.abstract_label.clone(),
                stoppable: x.stoppable
            },
            None => TraversalStateSpace {
                state: ProductStateSpace { s: 0, q: vec![] },
                a: TaskAction { a: 0, task: 0 },
                abstract_label: Default::default(),
                stoppable: false
            }
        }
    }

    pub fn traverse_n_steps(&self){
        //let mut step: u32 = 0;
        let mut new_state = Traversal::default();
        let mut finished: bool = false;
        let mut transition_choice = self.determine_choices(&self.initial.s, &self.initial.q);
        let mut current_state = TraversalStateSpace {
            state: ProductStateSpace { s: transition_choice.state.s, q: transition_choice.state.q.to_vec() },
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
            current_state = TraversalStateSpace{state: ProductStateSpace{s: new_state.data.s, q: new_state.data.q.to_vec()}, a: new_state.a, abstract_label: new_state.abstract_ap.clone(), stoppable: new_state.stoppable };
            // when the task has been completed we need to move onto the next task in the permutation
            if current_state.stoppable {
                // if all of the tasks are finished, then finished becomes true, otherwise
                if tasks_finished(&current_state.abstract_label) {
                    finished = true;
                } // choose the next action to go to
                else {
                    transition_choice = self.determine_choices(&current_state.state.s, &current_state.state.q);
                    let mut new_choice_state = TraversalStateSpace {
                        state: ProductStateSpace { s: transition_choice.state.s, q: transition_choice.state.q.to_vec() },
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
                            q: traversal.q.to_vec()
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
            Some(x) => Traversal{a: x.a, data: ProductStateSpace{ s: x.data.s, q: x.data.q.to_vec()}, p: x.p, abstract_ap: x.abstract_ap.clone(), stoppable: x.stoppable },
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
    p: f32,
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
    pub constraint: f32,
    pub reach_rewards: HashMap<u8, f32>,
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
                state: ProductStateSpace { s: x.s, q: x.q.to_vec() },
                a: TaskAction { a: x.a.a, task: x.a.task },
                abstract_label: x.abstract_label.clone(),
                stoppable: x.stoppable
            },
            None => TraversalStateSpace {
                state: ProductStateSpace { s: 0, q: vec![] },
                a: TaskAction { a: 0, task: 0 },
                abstract_label: Default::default(),
                stoppable: false
            }
        }
    }

    pub fn traverse_n_steps(&self){
        //let mut step: u32 = 0;
        let mut new_state = Traversal::default();
        let mut finished: bool = false;
        let mut transition_choice = self.determine_choices(&self.initial.s, &self.initial.q);
        println!("transition choice: {:?}", transition_choice);
        let mut current_state = TraversalStateSpace {
            state: ProductStateSpace { s: transition_choice.state.s, q: transition_choice.state.q.to_vec() },
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
            current_state = TraversalStateSpace{state: ProductStateSpace{s: new_state.data.s, q: new_state.data.q.to_vec()}, a: new_state.a, abstract_label: new_state.abstract_ap.clone(), stoppable: new_state.stoppable };
            // when the task has been completed we need to move onto the next task in the permutation
            if current_state.stoppable {
                // if all of the tasks are finished, then finished becomes true, otherwise
                if tasks_finished(&current_state.abstract_label){
                    finished = true;
                } // choose the next action to go to
                else {
                    transition_choice = self.determine_choices(&current_state.state.s, &current_state.state.q);
                    let mut new_choice_state = TraversalStateSpace {
                        state: ProductStateSpace { s: transition_choice.state.s, q: transition_choice.state.q.to_vec() },
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
                            q: traversal.q.to_vec()
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
            Some(x) => Traversal{a: x.a, data: ProductStateSpace{ s: x.data.s, q: x.data.q.to_vec()}, p: x.p, abstract_ap: x.abstract_ap.clone(), stoppable: x.stoppable },
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
                state_space_new.push(ProductStateSpace{ s: state.s, q: newq.to_vec()});
                for transition in self.transitions.iter().filter(|x| x.s == state.s && x.q == state.q) {
                    let initial = if transition.s == self.initial.s && transition.q == self.initial.q && *q == dfa.initial { true } else { false };
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
        self.initial = ProductStateSpace{s: self.initial.s, q: initial_dfa_vec};
        self.transitions = transitions_new;
        self.task_counter = self.task_counter + 1;
        self
    }
}
