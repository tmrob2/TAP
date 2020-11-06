use std::convert::TryFrom;
use rand::seq::SliceRandom;
use itertools::Itertools;

fn main() {

    let states: Vec<u32> = (0..4).collect();
    let transitions: Vec<Transition> = vec![
        Transition {
            s: 0,
            a: 1,
            s_prime: vec![TransitionPair{s: 1, p: 1.}]
        },
        Transition {
            s: 1,
            a: 1,
            s_prime: vec![TransitionPair{s: 0, p: 0.6}, TransitionPair{s: 2, p: 0.3}, TransitionPair{s:3, p:0.1}]
        },
        Transition {
            s: 1,
            a: 2,
            s_prime: vec![TransitionPair{s: 2, p: 0.5}, TransitionPair{s: 3, p: 0.5}]
        },
        Transition {
            s: 2,
            a: 1,
            s_prime: vec![TransitionPair{s: 2, p: 1.}]
        },
        Transition {
            s: 3,
            a: 1,
            s_prime: vec![TransitionPair { s: 3, p: 1. }]
        }
    ];
    for i in transitions.iter() {
        println!("s: {}, a: {}, s': {:?}", i.s, i.a, i.s_prime.iter());
    }

    let transitions2: Vec<Transition> = vec![
        Transition {
            s: 0,
            a: 1,
            s_prime: vec![TransitionPair{s: 0, p: 0.2}, TransitionPair{ s:1, p:0.8}],
        },
        Transition {
            s: 1,
            a: 1,
            s_prime: vec![TransitionPair{s:2, p:1.}]
        },
        Transition {
            s: 2,
            a: 1,
            s_prime: vec![TransitionPair{s: 3, p: 0.5}, TransitionPair{s: 4, p: 0.5}]
        },
        Transition {
            s: 2,
            a: 2,
            s_prime: vec![TransitionPair{s:4, p:1.}]
        }
    ];

    let x: Vec<i8> = vec![0,1,2,3,4];
    println!("{}", if x.iter().find(|&&x| x == 2) == Some(&2)  { true } else { false });

    println!("{}", x.contains(&2));

    let target = vec![2];
    let s0min = vec![3];
    println!("{:?}", value_iteration(&states, &transitions, 0.001, &target, &s0min));

    //let ap = vec!['a', 'b', 'c'];
    //let pset = powerset(&ap);
    //println!("{:?}", pset);

    // we can use vectors but we need to convert the vector to a slice with deref coercion
    println!("dra transition: {:?}", delta1(0, "initiate1"));
    // mdp labelling function
    println!("mdp labelling {:?}", mdp_labelling(1));
    // and then putting the two together
    let q: u32 = 0;
    println!("dra transition of an MDP labelling: {:?}", delta1(q, mdp_labelling(1)));

    // I guess we have to build the product up iteratively or recursively either way we call the func
    // one at a time.
    // Just like in the MDP we essentially created a dynamic list of transitions so we need to do
    // the same for the product DFA

    //println!("init product state space {:?}", init_prod_states);

    //let dfa_states: Vec<u32> = (0..4).collect();
    //let dfa2_states: Vec<u32> = (0..5).collect();
    let j_task: u32 = 0;

    let dfa1: DFA = DFA{
        states: vec![0,1,2,3],
        initial: 0u32,
        delta: delta1,
        rejected: vec![3u32],
        accepted: vec![2u32],
    };
    let dfa2 = DFA {
        states: vec![0,1,2,3,4],
        initial: 0,
        delta: delta2,
        rejected: vec![4u32],
        accepted: vec![3u32],
    };
    let mdp1 = MDP {
        states: vec![0,1,2,3,4],
        initial: 0,
        transitions: transitions2,
        labelling: mdp_labelling
    };
    // create an initial product MDP
    // first essentially we just want to modify the transitions
    let mut init_prod_states: Vec<ProductStateSpace> = Vec::new();
    for state in states.iter() {
        init_prod_states.push(ProductStateSpace{s: *state, q: Vec::new()})
    }
    let init_product_transitions = mdp1.convert_to_product();
    /*for ptrans in init_product_transitions.iter() {
        println!("{:?}", ptrans)
    }*/
    let init_prod_mdp = ProductMDP{
        states: init_prod_states,
        transitions: init_product_transitions,
        initial: ProductStateSpace{s: mdp1.initial, q: Vec::new()},
        labelling: mdp1.labelling,
    };

    let mut pmdp2 = product_mdp_v4(&dfa1, &init_prod_mdp, &j_task);
    // print the characteristics of the product mdp
    //println!("The initial state: {:?}", pmdp2.initial);
    //println!("The enumerated transitions");
    /*for ptransition in pmdp2.transitions.iter(){
        println!("{:?}", ptransition)
    }*/
    let tau = 3i8;
    let mdp1 = 1i8;
    //let mod_mdp2 = pmdp2.mod_prod_mdp(&tau, &mdp1);
    //for transition in mod_mdp2.transitions.iter(){
    //    println!("{:?}", transition);
    //}
    let j_task2: u32 = 1;
    pmdp2.labelling = mdp_labelling2;

    let pmdp3 = product_mdp_v4(&dfa2, &pmdp2, &j_task2);
    //println!("The initial state: {:?}", pmdp3.initial);
    //println!("The enumerated state space");
    /*for s in pmdp3.states.iter(){
        println!("{:?}", s)
    }*/
    //println!("The enumerated transitions");
    for ptransition in pmdp3.transitions.iter() {
        println!("{:?}", ptransition)
    }
    //let mod_mdp3 = pmdp3.mod_prod_mdp(&tau, &mdp1);
    /*for transition in mod_mdp3.transitions.iter() {
        println!("{:?}", transition)
    }*/
    //println!("Traversal testing; randomly generate a scheduler to traverse the product MDP");
    //mod_mdp2.reach_objective();
    //mod_mdp3.reach_objective();
}

// Consider moving the structures to another file, which will be cleaner to import as a module
#[derive(Debug)]
struct TransitionPair {
    s: u32, // need to type it this was so we can automatically reference arrays with states
    p: f32
}

#[derive(Debug)]
struct Transition {
    s: u32,
    a: i8,
    s_prime: Vec<TransitionPair>
}

#[derive(Debug)]
struct ProductTransitionPair {
    s: u32,
    p: f32,
    q: Vec<u32>,
    accepting: bool,
    rejecting: bool,
}


#[derive(Debug)]
struct ProductStateSpace {
    s: u32,
    q: Vec<u32>
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

#[derive(Debug)]
struct ProductTransition {
    s: u32,
    q: Vec<u32>,
    a: i8,
    s_prime: Vec<ProductTransitionPair>,
    self_loop: bool,
    first_rejected_info: Vec<RejectedStatus>,
    accepted: bool,
    rejected: bool,
    ap: Vec<String>,
}

#[derive(Debug)]
struct ModProductTransition {
    s: u32,
    q: Vec<u32>,
    a: i8,
    s_prime: Vec<ProductTransitionPair>,
    ap: Vec<string>,
}

struct Pair {
    q: u32,
    a: Vec<char>
}

#[derive(Debug)]
struct RejectedStatus {
    first_rejected: bool,
    state: u32,
    index: usize,
    p: f32,
}

struct DFA {
    states: Vec<u32>,
    initial: u32,
    delta: fn(u32, &str) -> u32,
    rejected: Vec<u32>,
    accepted: Vec<u32>
}

struct MDP {
    states: Vec<u32>,
    initial: u32,
    transitions: Vec<Transition>,
    labelling: fn(u32) -> &'static str,
}

#[derive(Debug)]
struct Traversal {
    a: i8, // some action
    data: ProductStateSpace, // some data which is a modified transition
    rejected: bool,
    accepting: bool,
    p: f32
}

impl Traversal {
    fn default () -> Traversal {
        Traversal{
            a: -1,
            data: ProductStateSpace {
                s: 0,
                q: Vec::new(),
            },
            rejected: false,
            accepting: false,
            p: 0.
        }
    }
}

impl MDP {
    fn convert_to_product(&self) -> Vec<ProductTransition> {
        let mut prod_transition: Vec<ProductTransition> = Vec::new();
        for transition in self.transitions.iter() {
            let mut sprime: Vec<ProductTransitionPair> = Vec::new();
            for s in transition.s_prime.iter(){
                sprime.push(ProductTransitionPair{
                    s: s.s,
                    p: s.p,
                    q: Vec::new(),
                    accepting: false,
                    rejecting: false,
                })
            }
            prod_transition.push(ProductTransition{
                s: transition.s,
                q: Vec::new(),
                a: transition.a,
                s_prime: sprime,
                self_loop: false,
                first_rejected_info: Vec::new(),
                accepted: false,
                rejected: false,
                ap: Vec::new(),
            })
        }
        prod_transition
    }
}

struct ProductMDP {
    states: Vec<ProductStateSpace>,
    transitions: Vec<ProductTransition>,
    initial: ProductStateSpace,
    labelling: fn(u32) -> &'static str,
}

impl ProductMDP {

    fn mod_labelling<'a>(&self, s: &u32, q: &u32, accepting: &bool, rejecting: &bool, justfail: bool, task_j: &u32, mdp_number: &i8, snew: bool) -> Vec<String> {
        // what is the label of the sub M

        if *q == 0u32 && snew == false {
            // then the task has not yet begun
            let word1 = format!("initial{}", task_j);
            let word2 = format!("Stoppable{}", mdp_number);
            vec![word1, word2]
        }
        else if *q== 0u32 && snew == true {
            let word1 = format!("initial{}", task_j);
            vec![word1]
        }
        else if *accepting == true && snew == false {
            let word1 = format!("complete{}", task_j);
            let word2 = format!("Stoppable{}", mdp_number);
            vec![word1, word2]

        }
        else if *accepting == true && snew == true {
            let word1 = format!("complete{}", task_j);
            vec![word1]
        }
        else if *rejecting == true && justfail == false && snew == false {
            let word1 = format!("fail{}", task_j);
            let word2 = format!("Stoppable{}", mdp_number);
            vec![word1, word2]
        }
        else if justfail == true && snew == false {
            let word1 = format!("justFail{}", task_j);
            let word2 = format!("Stoppable{}", mdp_number);
            vec![word1, word2]
        }
        else if justfail == true && snew == true {
            let word1 = format!("justFail{}", task_j);
            let word2 = format!("impossible{}", task_j);
            vec![word1, word2]
        }
        else {
            Vec::new()
        }
    }

    fn mod_prod_mdp(&self, tau: &i8, mdp_number: &i8) -> ModifiedProductMDP {
        // self loop modifications
        // first find the self loops using a filter then iterate through adding new states and transitions
        // if we consider the product MDP as immutable
        // what is the current length of the state space vector
        //let ss_size = self.states.len(); // state space size
        //let self_loops: Vec<&ProductTransition> = self.transitions.iter().filter(|x| x.self_loop == true).collect::<Vec<&ProductTransition>>();
        //println!("Total Transitions: {},\nself loops: {:?},\nno. of self loop transitions: {}", self.transitions.len(), self_loops, self_loops.len())
        let mut state_space_new: Vec<ProductStateSpace> = Vec::new();
        let mut counter = u32::try_from(self.states.len()).unwrap();
        let mut mod_prod_transitions: Vec<ModProductTransition> = Vec::new();

        for state in self.transitions.iter() {
            //println!("Original: {:?}", state);
            // we should be able to do all of the modifications in one loop
            if state.self_loop == true {
                state_space_new.push(ProductStateSpace{
                    s: state.s,
                    q: state.q.to_vec(),
                });
                state_space_new.push(ProductStateSpace{
                    s: counter,
                    q:state.q.to_vec(),
                });

                // so the state space has been modified and now we have to modify the transitions
                // also in this condition we want to add the labelling in for snew, if it requires
                // any changes.
                let mut sprime_new: Vec<ProductTransitionPair> = Vec::new();
                let mut snew_prime: Vec<ProductTransitionPair> = Vec::new();
                for sprime in state.s_prime.iter(){
                    if sprime.s == state.s && sprime.q == state.q {
                        // the self loop needs to be routed to snew
                        sprime_new.push(ProductTransitionPair{ s: counter, p: sprime.p, q: sprime.q.to_vec(), accepting: sprime.accepting, rejecting: sprime.rejecting});
                        snew_prime.push(ProductTransitionPair{ s: counter, p: sprime.p, q: state.q.to_vec(), accepting: sprime.accepting, rejecting: sprime.rejecting});
                    }
                    else {
                        // adjust the state space to include the normal transition state from the original state
                        state_space_new.push(ProductStateSpace{ s: sprime.s, q: sprime.q.to_vec() });
                        sprime_new.push(ProductTransitionPair{ s: sprime.s, p: sprime.p, q: sprime.q.to_vec(), accepting: sprime.accepting, rejecting: sprime.rejecting});
                        snew_prime.push(ProductTransitionPair{ s: sprime.s, p: sprime.p, q: sprime.q.to_vec(), accepting: sprime.accepting, rejecting: sprime.rejecting});
                    }
                }
                // the following are labelling functions for the product mdp
                let mut aptotal: HashSet<_> = HashSet::new();
                aptotal.extend(&state.ap);
                for (i,x) in state.q.iter().enumerate() {
                    let j = u32::try_from(i).unwrap();
                    let mut b: HashSet<&str> = HashSet::new();
                    let ap_new = self.mod_labelling(&state.s, x, &state.accepted, &state.rejected, false, &j, &mdp_number, false);
                    aptotal.extend(&ap_new);
                }
                let mut aptotal2: HashSet<_> = HashSet::new();
                aptotal2.extend(&state.ap);
                for (i,x) in state.q.iter().enumerate() {
                    let j = u32::try_from(i).unwrap();
                    let ap_new2 = self.mod_labelling(&state.s, x, &state.accepted, &state.rejected, false, &j, &mdp_number, true);
                    aptotal2.extend(&ap_new2);
                }

                mod_prod_transitions.push(ModProductTransition{
                    s: state.s,
                    q: state.q.to_vec(),
                    a: state.a,
                    s_prime: sprime_new,
                    ap: aptotal,
                });
                //println!("Transitions to snew: {:?}", transition_to_snew);
                mod_prod_transitions.push(ModProductTransition{
                    s: counter,
                    q: state.q.to_vec(),
                    a: state.a,
                    s_prime: snew_prime,
                    ap: aptotal2,
                });
                //println!("Transition from snew: {:?}", transition_from_snew);
            }
            else if state.first_rejected_info.len() > 0 {
                // this is the case where the state contains first time rejected info
                // i.e we will have to add a new state and then modify the transitions
                // here we also want to label the just fail state correctly, that is as just moving
                // to the first time failed
                state_space_new.push(ProductStateSpace {
                    s: state.s,
                    q: state.q.to_vec(),
                });
                for sprime in state.s_prime.iter() {
                    // if |state.first_rejected_info| > 0 then we know it contains two failure conditions, and if these
                    // are both just fails then they need to be labelled accordingly
                    for r in state.first_rejected_info.iter(){
                        let r_index = usize::try_from(r.index).unwrap();
                        if r_index == 0 {
                            if r.state == sprime.q[r_index] {
                                // this is a rejected state, with probability state.p the transition
                                // will go from state.s, q -> s_star=counter, q and the label will be justfail
                                let mut aptotal2: HashSet<_> = HashSet::new();
                                aptotal2.extend(&state.ap);
                                for (i,x) in state.q.iter().enumerate() {
                                    let j = u32::try_from(i).unwrap();
                                    let ap_new2 = self.mod_labelling(&state.s, x, &state.accepted, &state.rejected, true, &j, &mdp_number, false);
                                    aptotal2.extend(&ap_new2);
                                }
                                let mut aptotal: HashSet<_> = HashSet::new();
                                aptotal.extend(&state.ap);
                                for (i,x) in state.q.iter().enumerate() {
                                    let j = u32::try_from(i).unwrap();
                                    let ap_new = self.mod_labelling(&state.s, x, &state.accepted, &state.rejected, false, &j, &mdp_number, false);
                                    aptotal.extend(&ap_new);
                                }
                                state_space_new.push(ProductStateSpace{s: counter, q: state.q.to_vec()});
                                mod_prod_transitions.push(ModProductTransition{
                                    s: state.s,
                                    q: state.q.to_vec(),
                                    a: state.a,
                                    s_prime: vec![ProductTransitionPair{s: counter, p: sprime.p, q: sprime.q.to_vec(), accepting: sprime.accepting, rejecting: sprime.rejecting}],
                                    ap: aptotal,
                                });
                                //println!("Transition from s -> s*: {:?}", transition_s_star);
                                mod_prod_transitions.push(ModProductTransition{
                                    s: counter,
                                    q: sprime.q.to_vec(),
                                    a: *tau,
                                    s_prime: vec![ProductTransitionPair{s: sprime.s, p: 1., q: sprime.q.to_vec(), accepting: sprime.accepting, rejecting: sprime.rejecting}],
                                    ap: aptotal2
                                });
                                //println!("Transition from s* to fail: {:?}", transition_from_s_star);

                            }
                        }

                    }
                }
            }
            else {
                // this is the where there is neither a self loop or a just fail transition
                state_space_new.push(ProductStateSpace{s: state.s, q: state.q.to_vec()});
                let mut sprime_new: Vec<ProductTransitionPair> = Vec::new();
                for sprime in state.s_prime.iter(){
                    sprime_new.push(ProductTransitionPair{s: sprime.s, p: sprime.p, q: sprime.q.to_vec(), accepting: sprime.accepting, rejecting: sprime.rejecting});
                }
                let mut aptotal: HashSet<_> = HashSet::new();
                aptotal.extend(&state.ap);
                for (i,x) in state.q.iter().enumerate() {
                    let j = u32::try_from(i).unwrap();
                    let ap_new = self.mod_labelling(&state.s, x, &state.accepted, &state.rejected, false, &j, &mdp_number, false);
                    aptotal.union(&ap_new);
                }
                mod_prod_transitions.push(ModProductTransition{
                    s: state.s,
                    q: state.q.to_vec(),
                    a: state.a,
                    s_prime: sprime_new,
                    ap: aptotal
                });
                //println!("Neither self loop nor first rejected: {:?}", transition_normal);
            }
            counter = counter + 1;
        }

        let mod_mdp = ModifiedProductMDP {
            states: state_space_new,
            transitions: mod_prod_transitions,
            initial: ProductStateSpace{s: self.initial.s, q: self.initial.q.to_vec()},
            labelling: self.labelling,
            number: *mdp_number,
        };

        mod_mdp

    }
}

struct ModifiedProductMDP{
    states: Vec<ProductStateSpace>,
    transitions: Vec<ModProductTransition>,
    initial: ProductStateSpace,
    labelling: fn(u32) -> &'static str,
    number: i8,
}

impl ModifiedProductMDP {
    fn traversal(&self, input: &ProductStateSpace) -> Traversal {
        // this function is supposed to find paths through a graph
        // starting at the initial state we find a path through the product MDP
        let a_current: i8 = 1;
        let mut output: Vec<Traversal> = Vec::new();
        for x in self.transitions.iter().filter(|x| x.s == input.s && x.q == input.q) {
            let o = x.s_prime.choose(&mut rand::thread_rng());
            match o {
                Some(traversal) => output.push(
                    Traversal {
                        a: x.a,
                        data: ProductStateSpace{
                            s: traversal.s,
                            q: traversal.q.to_vec()
                        },
                        rejected: traversal.rejecting,
                        accepting: traversal.accepting,
                        p: traversal.p,
                    }
                ),
                None => println!("nothing")
            }
        }
        let result = output.choose(&mut rand::thread_rng());
        match result {
            Some(x) => Traversal{a: x.a, data: ProductStateSpace{ s: x.data.s, q: x.data.q.to_vec()}, rejected: x.rejected, accepting: x.accepting, p: x.p},
            None => Traversal::default()
        }
    }

    fn label(&self, s: &u32, q: &Vec<u32>) -> HashSet<&str> {
        let ap_return: HashSet<_> = HashSet::new();
        for transition in self.transitions.iter().filter(|x| x.s == *s && x.q == *q){
            //println!("{:?}", transition.ap);
            ap_return.union(&transition.ap);
        }
        ap_return
    }

    fn reach_objective(&self) {
        let mut done: bool = false;
        let mut current_state = ProductStateSpace{s: self.initial.s, q: self.initial.q.to_vec()};
        let mut new_state = Traversal::default();
        while !done {
            new_state = self.traversal(&current_state);
            if new_state.rejected == true {
                done = true;
            }
            println!("p((s:{},q{:?}) , a:{}, (s':{},q':{:?}))={}:", &current_state.s, &current_state.q, &new_state.a, &new_state.data.s, &new_state.data.q, &new_state.p);
            println!("label: {:?} -> {:?}", self.label(&current_state.s, &current_state.q), self.label(&new_state.data.s, &new_state.data.q));
            current_state = ProductStateSpace{s: new_state.data.s, q: new_state.data.q.to_vec()};
        }
    }
}

fn delta1(q:u32, a: &str) -> u32 {
    match (q, a) {
        (0, a) => if ["initiate1"].contains(&a) { 1} else { 0},
        (1, a) => if ["ready", "initiate1", "none"].contains(&a) { 1 } else if ["sprint1"].contains(&a) { 2 } else { 3 },
        (2, a) => 2,
        (3, a) => 3, // this should throw an error, therefore we will have to handle it
        _ => q,
    }
}

fn delta2(q:u32, a: &str) -> u32 {
    match (q, a) {
        (0, a) => if ["initiate2"].contains(&a) {1} else {0},
        (1, a) => if ["ready", "initiate2", "none"].contains(&a) {1} else if ["sprint2"].contains(&a) {2} else {4},
        (2, a) => if ["ready", "initiate2", "none"].contains(&a) {2} else if ["sprint2"].contains(&a) {3} else {4},
        (3, a) => 3,
        (4, a) => 4,
        _ => q, //
    }
}

fn delta3(q:u32, a: &str) -> u32 {
    match (q, a) {
        (0, a) => if ["initiate2"].contains(&a) {1} else {0},
        (1, a) => if ["ready", "initiate2", "none"].contains(&a) {1} else if ["sprint2"].contains(&a) {2} else {5},
        (2, a) => if ["ready", "initiate2", "none"].contains(&a) {2} else if ["sprint2"].contains(&a) {3} else {5},
        (3, a) => if ["ready", "initiate2", "none"].contains(&a) {3} else if ["sprint2"].contains(&a) {4} else {5},
        (4, a) => 4,
        (5, a) => 5,
        _ => q, //
    }
}

fn mdp_labelling<'a>(s: u32) -> &'a str {
    match s {
        0 => "none",
        1 => "initiate1",
        2 => "ready",
        3 => "sprint1",
        4 => "exit",
        _ => "error",
    }
}

fn mdp_labelling2<'a>(s: u32) -> &'a str {
    match s {
        0 => "none",
        1 => "initiate2",
        2 => "ready",
        3 => "sprint2",
        4 => "exit",
        _ => "error",
    }
}

fn mdp_labelling3<'a>(s: u32) -> &'a str {
    match s {
        0 => "none",
        1 => "initiate3",
        2 => "ready",
        3 => "sprint3",
        4 => "exit",
        _ => "error",
    }
}

fn product_mdp_v4(dfa: &DFA, pmdp: &ProductMDP, j_task: &u32) -> ProductMDP {
    //println!("New product");
    let mut new_product_states: Vec<ProductStateSpace> = Vec::new();
    let j = usize::try_from(*j_task).unwrap();
    for product_state in pmdp.states.iter(){
        for dfa_state in dfa.states.iter() {
            let mut p = ProductStateSpace {
                s: product_state.s,
                q: product_state.q.to_vec()
            };
            p.append_state(*dfa_state);
            new_product_states.push(p);
        }
    }
    //println!("{:?}",new_product_states);
    //println!("The start of the transitions");
    let mut new_product_transitions: Vec<ProductTransition> = Vec::new();
    //let mut product_transitions: Vec<>
    for state in new_product_states.iter(){

        if *j_task >= 1 {
            for transition in pmdp.transitions.iter().filter(|x| x.s == state.s && x.q[..] == state.q[..j] ) {
                //println!("state: {:?}, transition: {:?}", state, transition);
                let mut transition_to: Vec<ProductTransitionPair> = Vec::new();
                let mut self_loop: bool = false;
                let mut self_that: bool = false;
                let mut rejected_capture: Vec<RejectedStatus> = Vec::new();

                //println!("{:?}", transition.s_prime);
                for sprime in transition.s_prime.iter(){
                    for r in transition.first_rejected_info.iter(){
                        if r.state == sprime.q[r.index]{
                            rejected_capture.push(RejectedStatus{first_rejected: r.first_rejected, state: r.state, p: r.p , index: r.index});
                            //println!("Q: {:?}, Original rejected: {:?}", sprime.q, r)
                        }
                    }
                    let label = (pmdp.labelling)(sprime.s); // The labelling function always = L(s') i.e. L(s, qbar) = L(s')
                    let qprime: u32 = (dfa.delta)(state.q[j], label); // this references the product function directly and is therefore always relevant
                    let mut qprime_new: Vec<u32> = sprime.q[..j].to_vec();
                    qprime_new.push(qprime);

                    // if q' is one of the rejected states
                    if dfa.rejected.contains(&qprime) && !dfa.rejected.contains(&state.q[j]){
                        rejected_capture.push(RejectedStatus {
                            first_rejected: true,
                            state: qprime,
                            p: sprime.p,
                            index: j,
                        });
                    }

                    if sprime.s == state.s && qprime_new == state.q { self_loop = true }
                    if sprime.s != state.s && qprime_new != state.q { self_that = true }
                    //println!("s': {}, s: {}, s==s': {}", sprime.s,state.s, sprime.s==state.s);
                    //println!("q': {:?}, q: {:?}, q==q': {}, self_loop: {}, self_that: {}", state.q, qprime_new, state.q==qprime_new, self_loop, self_that);
                    //println!("(s,q), a, (s',q') = p: ({},{:?}),{},({},{:?}) = {}, label: {}", state.s, state.q, transition.a, sprime.s, qprime_new, sprime.p, label);
                    transition_to.push(ProductTransitionPair{s: sprime.s, p: sprime.p, q: qprime_new, accepting: if dfa.accepted.contains(&qprime) { true } else { false }, rejecting: if dfa.rejected.contains(&qprime) { true } else { false }});
                }

                let mut snew_cond: bool = false;
                if self_loop == true && self_that == true { snew_cond = true; }
                //println!("self loop: {}, that loop: {}, snew: {}", self_loop, self_that, snew_cond);
                //let ap_new: HashSet<_> = HashSet::new();
                let state_label = (pmdp.labelling)(state.s);
                let mut v: HashSet<_> = vec![state_label].into_iter().collect();
                v.extend(&transition.ap);
                new_product_transitions.push(ProductTransition {
                    s: state.s,
                    q: state.q.to_vec(),
                    a: transition.a,
                    s_prime: transition_to,
                    self_loop: snew_cond,
                    first_rejected_info: rejected_capture,
                    accepted: if dfa.accepted.contains(&state.q[j]) { true } else { false },
                    rejected: if dfa.rejected.contains(&state.q[j]) {true} else {false},
                    ap: v
                });
            }
        }
        else {
            for transition in pmdp.transitions.iter().filter(|x| x.s == state.s ) {
            //println!("{:?}", transition);
                let mut transition_to: Vec<ProductTransitionPair> = Vec::new();
                let mut self_loop: bool = false;
                let mut self_that: bool = false;
                let mut rejected_capture: Vec<RejectedStatus> = Vec::new();

                //println!("{:?}", transition.s_prime);
                for sprime in transition.s_prime.iter(){
                    let label = (pmdp.labelling)(sprime.s); // The labelling function always = L(s') i.e. L(s, qbar) = L(s')
                    let qprime: u32 = (dfa.delta)(state.q[j], label); // this references the product function directly and is therefore always relevant
                    let mut qprime_new: Vec<u32> = sprime.q[..j].to_vec();
                    qprime_new.push(qprime);

                    // if q' is one of the rejected states
                    if dfa.rejected.contains(&qprime) && !dfa.rejected.contains(&state.q[j]){
                        rejected_capture.push(RejectedStatus {
                            first_rejected: true,
                            state: qprime,
                            p: sprime.p,
                            index: j
                        });
                    }

                    if sprime.s == state.s && qprime_new == state.q { self_loop = true }
                    if sprime.s != state.s && qprime_new != state.q { self_that = true }
                    //println!("s': {}, s: {}, s==s': {}", sprime.s,state.s, sprime.s==state.s);
                    //println!("q': {:?}, q: {:?}, q==q': {}, self_loop: {}, self_that: {}", state.q, qprime_new, state.q==qprime_new, self_loop, self_that);
                    //println!("(s,q), a, (s',q') = p: ({},{:?}),{},({},{:?}) = {}, label: {}", state.s, state.q, transition.a, sprime.s, qprime_new, sprime.p, label);

                    transition_to.push(ProductTransitionPair{s: sprime.s, p: sprime.p, q: qprime_new, accepting: if dfa.accepted.contains(&qprime) { true } else { false }, rejecting: if dfa.rejected.contains(&qprime) { true } else { false }});
                }

                let mut snew_cond: bool = false;
                if self_loop == true && self_that == true { snew_cond = true; }
                //println!("self loop: {}, that loop: {}, snew: {}", self_loop, self_that, snew_cond);
                //let ap_new: HashSet<_> = HashSet::new();
                let state_label = (pmdp.labelling)(state.s);
                let mut v: HashSet<_> = vec![state_label].into_iter().collect();
                v.extend(&transition.ap);
                new_product_transitions.push(ProductTransition {
                    s: state.s,
                    q: state.q.to_vec(),
                    a: transition.a,
                    s_prime: transition_to,
                    self_loop: snew_cond,
                    first_rejected_info: rejected_capture,
                    accepted: if dfa.accepted.contains(&state.q[j]) { true } else { false },
                    rejected: if dfa.rejected.contains(&state.q[j]) {true} else {false},
                    ap: v,
                });
            }
        }
    }

    let mut pinitial_q: Vec<u32> = pmdp.initial.q.to_vec();
    pinitial_q.push(dfa.initial);

    let initial = ProductStateSpace{s: pmdp.initial.s, q: pinitial_q};

    ProductMDP {
        states: new_product_states,
        transitions: new_product_transitions,
        initial: initial,
        labelling: pmdp.labelling,
    }
}

fn value_iteration(states:&Vec<u32>, transitions: &Vec<Transition>, epsilon: f32, target: &Vec<u32>, s0min: &Vec<u32>) {
    let mut delta: f32 = 1.;
    let l = states.len();
    let mut x: Vec<f32> = vec![0.; l];
    let mut xprime: Vec<f32> = vec![0.; l];
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
    println!("calculable states: {:?}", r);
    while delta > epsilon {
        for s in r.iter() {
            // The next part of this problem is filtering dynamic arrays of custom structs
            {
                let s_index= usize::try_from(*s).unwrap();
                //println!("s_index: {:?}", s_index);
                let mut choose_arr: Vec<f32> = Vec::new();
                for n in transitions.iter().filter(|x| x.s == *s) {
                    //println!("{:?}", n);
                    // we need the transitions from
                    let mut sumarr: Vec<f32> = Vec::new();
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
        println!("x' = {:?}", xprime);
        let mut delta_v: Vec<f32> = vec![0.; x.len()];
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

}