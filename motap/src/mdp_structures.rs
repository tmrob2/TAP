use std::collections::{HashSet, HashMap};
use rand::seq::SliceRandom;
use std::convert::TryFrom;
use permutohedron::LexicalPermutation;

#[derive(Debug)]
pub struct TransitionPair {
    pub s: u32, // need to type it this was so we can automatically reference arrays with states
    pub p: f32
}

#[derive(Debug)]
pub struct Transition {
    pub s: u32,
    pub a: i8,
    pub s_prime: Vec<TransitionPair>
}

#[derive(Debug, Clone)]
pub struct ProductTransitionPair {
    pub s: u32,
    pub p: f32,
    pub q: Vec<u32>,
    pub abstract_label: HashMap<u8, TaskProgress>,
    //pub accepting: bool,
    //pub rejecting: bool,
    pub ap: String // I am wondering now whether this is just a string
}

#[derive(Debug, Clone)]
pub struct ProductStateSpace {
    pub s: u32,
    pub q: Vec<u32>
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

#[derive(Debug, Copy, Clone)]
pub struct TaskAction {
    pub a: i8,
    pub task: u8,
}

#[derive(Debug, Clone)]
pub struct ProductTransition {
    pub s: u32,
    pub q: Vec<u32>,
    pub a: TaskAction,
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

impl MDP {

    pub fn get_abstract_label(q: Option<&u32>, qprime: &u32, dfa: &DFA) -> (TaskProgress,bool) {
        match q {
            Some(x) => {
                if dfa.accepted.contains(qprime) {
                    (TaskProgress::Finished, true) }
                else if dfa.rejected.contains(qprime) && dfa.rejected.contains(x) {
                    (TaskProgress::Failed, true)
                } else if dfa.rejected.contains(qprime) && !dfa.rejected.contains(x) {
                    (TaskProgress::JustFailed, true)
                } else if dfa.initial == *qprime {
                    (TaskProgress::Initial, false)
                } else {
                    (TaskProgress::InProgress, false)
                }
            },
            None => {
                if dfa.accepted.contains(qprime) {
                    (TaskProgress::Finished, true) }
                else if dfa.rejected.contains(qprime) {
                    (TaskProgress::Failed, true)
                } else if dfa.initial == *qprime {
                    (TaskProgress::Initial, false)
                } else {
                    (TaskProgress::InProgress, false)
                }
            }
        }
    }

    pub fn initial_product_mdp<'a>(&'a self, dfa: &'a DFA<'a>, empty_container: &'a mut ProductMDP<'a>) -> &'a mut ProductMDP<'a> {
        // create a new state space based on the mdp and the dfa
        let mut state_space_new: Vec<ProductStateSpace> = Vec::new();
        let mut transitions_new: Vec<ProductTransition> = Vec::new();
        for state in self.states.iter() {
            for q in dfa.states.iter() {
                state_space_new.push(ProductStateSpace{ s: *state, q: vec![*q] });
                for transition in self.transitions.iter().filter(|x| x.s == *state) {
                    let mut sprimes: Vec<ProductTransitionPair> = Vec::new();
                    for sprime in transition.s_prime.iter() {
                        // name the dfa state that we are going to with the transition label
                        let qprime: u32 = (dfa.delta)(*q, (self.labelling)(sprime.s));
                        // determine if it is a self loop or not
                        let mut task_progress: HashMap<u8, TaskProgress> = HashMap::new();
                        let (progress_value, _) = MDP::get_abstract_label(Some(&q), &qprime, &dfa);
                        task_progress.insert(0, progress_value);
                        sprimes.push(ProductTransitionPair{
                            s: sprime.s,
                            p: sprime.p,
                            q: vec![qprime],
                            abstract_label: task_progress,
                            ap: format!("{}",(self.labelling)(sprime.s)),
                        });
                    }

                    let mut state_progress: HashMap<u8, TaskProgress> = HashMap::new();
                    let (state_progress_value, stoppable_value) = MDP::get_abstract_label(None, q, &dfa);
                    state_progress.insert(0, state_progress_value);
                    transitions_new.push(ProductTransition{
                        s: *state,
                        q: vec![*q],
                        a: TaskAction {a: transition.a, task: 0},
                        s_prime: sprimes,
                        abstract_label: state_progress,
                        stoppable: if transition.s == self.initial && *q == dfa.initial { true } else {stoppable_value },
                        ap: format!("{}", (self.labelling)(*state)),
                    })
                }
            }
        }
        empty_container.dfa_delta.insert(0u8, &dfa.delta);
        empty_container.states = state_space_new;
        empty_container.transitions = transitions_new;
        empty_container.initial = ProductStateSpace{ s: self.initial, q: vec![dfa.initial] };
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
    pub ap: Vec<String>
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

    pub fn generate_mod_product(&mut self, pmdp: &'a ProductMDP) -> &'a ModifiedProductMDP {
        // There are three steps to generating a modified product MDP
        // 1. A new state is added for self loops, this is so that we can clearly identify if the
        //    task has begun
        // 2. A state is added for justFail processes, this is so that we can conduct reachability
        //    analysis
        // 3. All other transitions, and states need to be inherited from the base local product

        // self loops definitions for A'(s,q) subset A(s,q) such that a in A'(s,q) iff
        // P'((s,q), a, (s,q)) > 0 && P'((s,q), a, (s'q'))>0 for some (s',q') != (s,q)
        let mut mod_prod_state_space: Vec<ProductStateSpace> = Vec::new();
        let mut mod_prod_transitions: Vec<ModProductTransition> = Vec::new();
        let mut counter: u32 = u32::try_from(pmdp.states.len()).unwrap();
        for transition in pmdp.transitions.iter() {
            // that is for self loop identification, there needs to be a edge from (s,q) -> (s,q)
            // and there needs to be an edge from (s,q) -> (s',q') where (s',q') != (s,q)
            let loop_cond: bool = transition.s_prime.iter().any(|x| x.s == transition.s);
            let no_loop_cond: bool = transition.s_prime.iter().any(|x| x.s != transition.s);
            if loop_cond == true && no_loop_cond == true {
            //if transition.self_loop == true {
                // add a new state to the state space, this state will be snew
                mod_prod_state_space.push(ProductStateSpace{ s: counter, q: transition.q.to_vec() });
                //println!("Self loop");
                // edit the current transition so that is goes to the new state, and then add a transition from the
                // new state to the original sprime state
                // Another thing to consider is that this must be a self loop, it cannot be anything else because it
                // satisfies the definition, therefore all of the sprimes must be edited
                let mut sprimes_current: Vec<ProductTransitionPair> = Vec::new();
                let mut sprimes_snew: Vec<ProductTransitionPair> = Vec::new();
                for sprime in transition.s_prime.iter(){
                    // if (s,q) != (s',q') =>  copy, and copy and edit from new state
                    if sprime.s != transition.s && sprime.q != transition.q {
                        // this is not a loop
                        sprimes_current.push(sprime.clone()); // original this will be from snew to s'
                        sprimes_snew.push(sprime.clone());
                    } else {
                        // these should only be self loops, but there may be circumstances where we have been tricked somewhere
                        // in the creation of the product MDP, something to watch out for...
                        // in the case where is it a loop, both versions of the transitions will be edited, and the abstract AP will be edited
                        let mut abstract_ap_new = sprime.abstract_label.clone();
                        let task_progress_new = ModifiedProductMDP::mod_abstract_ap(abstract_ap_new.get(&transition.a.task));
                        abstract_ap_new.remove(&transition.a.task);
                        abstract_ap_new.insert(transition.a.task, task_progress_new);
                        let sprime_loop_val = ProductTransitionPair{
                            s: counter,
                            p: sprime.p,
                            q: sprime.q.to_vec(),
                            abstract_label: abstract_ap_new,
                            ap: "".to_string() // starting to question if we need a string interpretation of the abstract ap
                        };
                        sprimes_current.push(sprime_loop_val.clone());
                        sprimes_snew.push(sprime_loop_val.clone());
                    }
                }
                // this is the current transition
                mod_prod_transitions.push(ModProductTransition{
                    s: transition.s,
                    q: transition.q.to_vec(),
                    a: TaskAction {a: transition.a.a, task: transition.a.task},
                    s_prime: sprimes_current,
                    abstract_label: transition.abstract_label.clone(),
                    ap: Vec::new(),
                    stoppable: transition.stoppable
                });
                // this is the transitions for snew
                mod_prod_transitions.push(ModProductTransition{
                    s: counter,
                    q: transition.q.to_vec(),
                    a: TaskAction {a: transition.a.a, task: transition.a.task},
                    s_prime: sprimes_snew,
                    abstract_label: transition.abstract_label.clone(),
                    ap: Vec::new(),
                    stoppable: false
                });
                counter = counter + 1;

            }
            else if loop_cond == true && no_loop_cond == false {
                // there is only a loop to itself, and nothing else, therefore no progress is possible
                //println!("pure loop")
                //println!("Pure loop: {:?}", transition);
                mod_prod_state_space.push(ProductStateSpace{ s: transition.s, q: transition.q.to_vec() });

                mod_prod_transitions.push(ModProductTransition{
                    s: transition.s,
                    q: transition.q.to_vec(),
                    a: TaskAction {a: transition.a.a, task: transition.a.task},
                    s_prime: transition.s_prime.to_vec(),
                    ap: vec![],
                    abstract_label: transition.abstract_label.clone(),
                    stoppable: transition.stoppable
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
                mod_prod_state_space.push(ProductStateSpace{s: transition.s, q: transition.q.to_vec()});
                if current_state_abstract_ap == true {
                    let mut sprime_current: Vec<ProductTransitionPair> = Vec::new();
                    //println!("Orginal: ({},{:?}) -> ({:?})", transition.s, transition.q, transition.s_prime);
                    for sprime in transition.s_prime.iter() {
                        // if there is an s' which has a task member of justFail, then we add a new state, called s*
                        match sprime.abstract_label.get(current_task) {
                            Some(x) => match x {
                                TaskProgress::JustFailed => {
                                    //println!("justFail transition found, state: ({}, {:?}), {}, ({}, {:?}) task: {}",
                                    //         transition.s, transition.q, transition.a.a, sprime.s, sprime.q, current_task);
                                    mod_prod_state_space.push(ProductStateSpace{ s: counter, q: transition.q.to_vec() });
                                    // a product transition to s*, then a product transition from s* to the original s',
                                    // which is just a copy of the original transition
                                    let mut new_abstract_label: HashMap<u8, TaskProgress> = sprime.abstract_label.clone();
                                    new_abstract_label.remove(&transition.a.task);
                                    new_abstract_label.insert(transition.a.task, TaskProgress::Failed);
                                    let sstar_sprime = ProductTransitionPair{
                                        s: sprime.s,
                                        p: 1.,
                                        q: sprime.q.to_vec(),
                                        abstract_label: new_abstract_label,
                                        ap: "".to_string(),
                                    };
                                    mod_prod_transitions.push(ModProductTransition{
                                        s: counter,
                                        q: sprime.q.to_vec(),
                                        a: TaskAction {a: transition.a.a, task: transition.a.task},
                                        s_prime: vec![sstar_sprime],
                                        ap: vec![format!("justFail{}", transition.a.task)],
                                        abstract_label: sprime.abstract_label.clone(),
                                        stoppable: true
                                    });

                                    sprime_current.push(ProductTransitionPair{
                                        s: counter,
                                        p: sprime.p,
                                        q: sprime.q.to_vec(),
                                        abstract_label: sprime.abstract_label.clone(),
                                        ap: "".to_string()
                                    });
                                    //println!("Added transition: ({},{:?}) -> ({},{:?}) -> ({}, {:?})", transition.s, transition.q.to_vec(), counter, sprime.q, sprime.s, sprime.q);
                                    counter = counter + 1;
                                },
                                _ => {
                                    // this is the case where we do not see a transition to justFail, and therefore we just copy the transition
                                    sprime_current.push(ProductTransitionPair{
                                        s: sprime.s,
                                        p: sprime.p,
                                        q: sprime.q.to_vec(),
                                        abstract_label: sprime.abstract_label.clone(),
                                        ap: "".to_string()
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
                        s_prime: sprime_current,
                        ap: vec![format!("!justFail{}", transition.a.task)],
                        abstract_label: transition.abstract_label.clone(),
                        stoppable: transition.stoppable
                    })

                } else {
                    // we are in an end condition and we should just copy over the transition
                    mod_prod_state_space.push(ProductStateSpace{s: transition.s, q: transition.q.to_vec()});
                    mod_prod_transitions.push(ModProductTransition{
                        s: transition.s,
                        q: transition.q.to_vec(),
                        a: TaskAction {a: transition.a.a, task: transition.a.task},
                        s_prime: transition.s_prime.to_vec(),
                        ap: vec![],
                        abstract_label: transition.abstract_label.clone(),
                        stoppable: transition.stoppable
                    })
                }
            }
        }

        self.states = mod_prod_state_space;
        self.transitions = mod_prod_transitions;
        self.task_counter = pmdp.task_counter;
        self.initial = pmdp.initial.clone();
        self

    }

    pub fn traverse_n_steps(&self){
        //let mut step: u32 = 0;
        let mut new_state = Traversal::default();
        let mut finished: bool = false;
        let task_perms: Vec<Vec<u8>> = generate_task_permuatations(self.task_counter);
        // choose a permutation form the vector of permutations
        let task_int = task_perms.choose(&mut rand::thread_rng()).unwrap();
        println!("Random task vector: {:?}", task_int);
        let mut task_iter = task_int.iter();
        let mut current_task = task_iter.next().unwrap();
        let mut current_state = TraversalStateSpace{state: ProductStateSpace{s: self.initial.s, q: self.initial.q.to_vec()}, a: TaskAction{a: 1, task: *current_task}};
        let mut task: bool = false;
        let mut debug_just_fail: bool = false;
        println!("First state: {:?}", current_state);
        while !finished {
            while !task {
                new_state = self.traversal(&current_state);
                print!("p((s:{},q{:?}) , a:{:?}, (s':{},q':{:?}))={}: ", &current_state.state.s, &current_state.state.q, &new_state.a, &new_state.data.s, &new_state.data.q, &new_state.p);
                println!("abstract label: {:?} -> {:?}", self.label(&current_state.state.s, &current_state.state.q, &current_state.a), self.label(&new_state.data.s, &new_state.data.q, &new_state.a));
                match new_state.abstract_ap.get(current_task) {
                    Some(x) => match x {
                        //TaskProgress::JustFailed => {task = true; debug_just_fail = true},
                        TaskProgress::Finished => {task = true},
                        TaskProgress::Failed => {task = true},
                        _ => {}
                    },
                    None => {println!("On call of abstract AP, returned None, current task: {}", current_task); task = false}
                }
                current_state = TraversalStateSpace{state: ProductStateSpace{s: new_state.data.s, q: new_state.data.q.to_vec()}, a: new_state.a};
            }
            // when the task has been completed we need to move onto the next task in the permutation
            match task_iter.next() {
                Some(x) => {current_task = x; println!("next task: {}", current_task); task=false},
                None => {finished=true; println!("finished")}
            }
            current_state = TraversalStateSpace{state: ProductStateSpace{s: current_state.state.s, q: current_state.state.q.to_vec()}, a: TaskAction{a: current_state.a.a, task: *current_task}};
        }
    }

    fn traversal(&self, input: &TraversalStateSpace) -> Traversal {
        // this function is supposed to find paths through a graph
        // starting at the initial state we find a path through the product MDP
        //let a_current: i8 = 1;
        let mut output: Vec<Traversal> = Vec::new();
        for x in self.transitions.iter().filter(|x| x.s == input.state.s && x.q == input.state.q && x.a.a == input.a.a && x.a.task == input.a.task) {
            println!("State found: ({},{:?},action:{:?}, (s',q'): {:?}, stoppable: {}", x.s, x.q, x.a, x.s_prime, x.stoppable);
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
                    }
                ),
                None => println!("nothing")
            }
        }
        let result = output.choose(&mut rand::thread_rng());
        match result {
            Some(x) => Traversal{a: x.a, data: ProductStateSpace{ s: x.data.s, q: x.data.q.to_vec()}, p: x.p, abstract_ap: x.abstract_ap.clone()},
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
}

#[derive(Debug)]
struct Traversal {
    a: TaskAction, // some action
    data: ProductStateSpace, // some data which is a modified transition
    p: f32,
    abstract_ap: HashMap<u8, TaskProgress>,
}

#[derive(Debug)]
pub struct TraversalStateSpace {
    state: ProductStateSpace,
    a: TaskAction,
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
    pub s_prime: Vec<ProductTransitionPair>,
    pub ap: Vec<String>,
    pub abstract_label: HashMap<u8, TaskProgress>,
    pub stoppable: bool,
}

pub fn generate_task_permuatations(no_tasks: u8) -> Vec<Vec<u8>>{
    let mut task_perms: Vec<Vec<u8>> = Vec::new();
    let mut tasks_ordered: Vec<u8> = (0..no_tasks).collect();
    let task_heap = permutohedron::Heap::new(&mut tasks_ordered);
    for task in task_heap {
        task_perms.push(task.clone())
    }
    task_perms
}

impl <'a> ProductMDP <'a> {

    pub fn traverse_n_steps(&self){
        //let mut step: u32 = 0;
        let mut new_state = Traversal::default();
        let mut finished: bool = false;
        let task_perms: Vec<Vec<u8>> = generate_task_permuatations(self.task_counter);
        // choose a permutation form the vector of permutations
        let task_int = task_perms.choose(&mut rand::thread_rng()).unwrap();
        println!("Random task vector: {:?}", task_int);
        let mut task_iter = task_int.iter();
        let mut current_task = task_iter.next().unwrap();
        let mut current_state = TraversalStateSpace{state: ProductStateSpace{s: self.initial.s, q: self.initial.q.to_vec()}, a: TaskAction{a: 1, task: *current_task}};
        let mut task: bool = false;
        println!("First state: {:?}", current_state);
        while !finished {
            while !task {
                new_state = self.traversal(&current_state);
                print!("p((s:{},q{:?}) , a:{:?}, (s':{},q':{:?}))={}: ", &current_state.state.s, &current_state.state.q, &new_state.a, &new_state.data.s, &new_state.data.q, &new_state.p);
                println!("abstract label: {:?} -> {:?}", self.label(&current_state.state.s, &current_state.state.q, &current_state.a), self.label(&new_state.data.s, &new_state.data.q, &new_state.a));
                match new_state.abstract_ap.get(current_task) {
                    Some(x) => match x {
                        TaskProgress::JustFailed => {task = true},
                        TaskProgress::Finished => {task = true},
                        TaskProgress::Failed => {task = true},
                        _ => {}
                    },
                    None => {println!("On call of abstract AP, returned None, current task: {}", current_task); task = false}
                }

                current_state = TraversalStateSpace{state: ProductStateSpace{s: new_state.data.s, q: new_state.data.q.to_vec()}, a: new_state.a};
            }
            // when the task has been completed we need to move onto the next task in the permutation
            match task_iter.next() {
                Some(x) => {current_task = x; println!("next task: {}", current_task); task=false},
                None => {finished=true; println!("finished")}
            }
            current_state = TraversalStateSpace{state: ProductStateSpace{s: current_state.state.s, q: current_state.state.q.to_vec()}, a: TaskAction{a: current_state.a.a, task: *current_task}};
        }
    }

    fn traversal(&self, input: &TraversalStateSpace) -> Traversal {
        // this function is supposed to find paths through a graph
        // starting at the initial state we find a path through the product MDP
        //let a_current: i8 = 1;
        let mut output: Vec<Traversal> = Vec::new();
        for x in self.transitions.iter().filter(|x| x.s == input.state.s && x.q == input.state.q && x.a.a == input.a.a && x.a.task == input.a.task) {
            println!("State found: ({},{:?},action:{:?}, (s',q'): {:?}, stoppable: {}", x.s, x.q, x.a, x.s_prime, x.stoppable);
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
                    }
                ),
                None => println!("nothing")
            }
        }
        let result = output.choose(&mut rand::thread_rng());
        match result {
            Some(x) => Traversal{a: x.a, data: ProductStateSpace{ s: x.data.s, q: x.data.q.to_vec()}, p: x.p, abstract_ap: x.abstract_ap.clone()},
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
                        let (task_label, _) = MDP::get_abstract_label(Some(q), &qprime, dfa);
                        task_progress.insert(*task, task_label);

                        sprimes.push(ProductTransitionPair{
                            s: sprime.s,
                            p: sprime.p,
                            q: qprimes,
                            abstract_label: task_progress,
                            ap: sprime.ap.to_string()
                        });

                        // new transition
                        let qprime2: u32 = (dfa.delta)(*q, (new_labelling)(sprime.s));
                        let mut qprimes2: Vec<u32> = transition.q.to_vec();
                        qprimes2.push(qprime2);

                        let mut task_progress2: HashMap<u8, TaskProgress> = transition.abstract_label.clone();
                        let (task_label2, stoppable) = MDP::get_abstract_label(Some(q), &qprime2, dfa);
                        task_progress2.insert(*task, task_label2);

                        sprimes_new.push(ProductTransitionPair{
                            s: sprime.s,
                            p: sprime.p,
                            q: qprimes2,
                            abstract_label: task_progress2,
                            ap: format!("{}", (new_labelling)(sprime.s))
                        });
                    }
                    let mut state_label: HashMap<u8, TaskProgress> = transition.abstract_label.clone();
                    let (state_label_value, stoppable_value) = MDP::get_abstract_label(None, q, dfa);
                    state_label.insert(*task, state_label_value);
                    transitions_new.push(ProductTransition{
                        s: state.s,
                        q: newq.to_vec() ,
                        a: TaskAction { a: transition.a.a, task: transition.a.task },
                        s_prime: sprimes,
                        abstract_label: state_label.clone(),
                        stoppable: if initial { true } else { transition.stoppable },
                        ap: transition.ap.to_string(),
                    });
                    transitions_new.push(ProductTransition{
                        s: state.s,
                        q: newq.to_vec() ,
                        a: TaskAction { a: transition.a.a, task: *task },
                        s_prime: sprimes_new,
                        abstract_label: state_label.clone(),
                        stoppable: if initial { true } else { transition.stoppable },
                        ap: format!("{}", (new_labelling)(state.s)),
                    })
                }
            }
        }
        let mut initial_dfa_vec: Vec<u32> = self.initial.q.to_vec();
        initial_dfa_vec.push(dfa.initial);
        self.states = state_space_new;
        self.initial = ProductStateSpace{s: self.initial.s, q: initial_dfa_vec};
        self.transitions = transitions_new;
        self.task_counter = self.task_counter + 1;
        self
    }

}
