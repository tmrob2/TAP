mod scratch;

use std::convert::TryFrom;

fn main() {
    //println!("Hello, world!");
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
    let mut init_prod_states: Vec<ProductStateSpace> = Vec::new();
    for state in states.iter() {
        init_prod_states.push(ProductStateSpace{s: *state, q: Vec::new()})
    }
    println!("init product state space {:?}", init_prod_states);

    let dfa_states: Vec<u32> = (0..4).collect();
    //let dfa2_states: Vec<u32> = (0..5).collect();
    let j_task: u32 = 0;

    let dfa1: DFA = DFA{
        states: vec![0,1,2,3],
        initial: 0u32,
        delta: delta1,
        rejected: vec![3u32],
        accepted: vec![2u32],
    };
    let mdp1 = MDP {
        states: vec![0,1,2,3,4],
        initial: 0,
        transitions: transitions2,
        labelling: mdp_labelling
    };
    //let (a1, b1) = product_mdp_v2(&dfa1, mdp_labelling, &mut init_prod_states, &transitions2, &j_task);
    let (a1, b1) = product_mdp_v3(&dfa1, &mdp1, &mut init_prod_states, &j_task);

    for i in b1.iter() {
        println!("prod f (dfa struct): s: {}, q: {:?}, a: {}, s':{:?}, contains loop: {}", i.s, i.q.to_vec(), i.a, i.s_prime, i.self_loop)
    }
    //let j_task2: u32 = 1;
    //let (mut a2, b2) = product_mdp(delta2, mdp_labelling2, &mut a, &dfa2_states, &transitions, &j_task2);

    //let j_task3: u32 = 2;
    //let (mut a3, b3) = product_mdp(delta3, mdp_labelling3, &mut a2, &dfa2_states, &transitions, &j_task3);
}

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
}

#[derive(Debug)]
struct ProductTransition {
    s: u32,
    q: Vec<u32>,
    a: i8,
    s_prime: Vec<ProductTransitionPair>,
    self_loop: bool,
    to_rejected: bool,
}

struct Pair {
    q: u32,
    a: Vec<char>
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

fn product_mdp_v3(dfa: &DFA, mdp: &MDP, product_mdp_states: &mut Vec<ProductStateSpace>, j_task: &u32)
    -> (Vec<ProductStateSpace>, Vec<ProductTransition>) {
    let mut new_product_states: Vec<ProductStateSpace> = Vec::new();
    let j = usize::try_from(*j_task).unwrap();
    for product_state in product_mdp_states.iter_mut(){
        for dfa_state in dfa.states.iter() {
            let mut p = ProductStateSpace {
                s: product_state.s,
                q: product_state.q.to_vec()
            };
            p.append_state(*dfa_state);
            new_product_states.push(p);
        }
    }
    println!("{:?}",new_product_states);

    let mut new_product_transitions: Vec<ProductTransition> = Vec::new();
    //let mut product_transitions: Vec<>
    for state in new_product_states.iter(){
        for transition in mdp.transitions.iter().filter(|x| x.s == state.s) {
            let mut transition_to: Vec<ProductTransitionPair> = Vec::new();
            let mut self_loop: bool = false;
            let mut self_that: bool = false;

            //println!("{:?}", transition.s_prime);
            for sprime in transition.s_prime.iter(){
                let label = (mdp.labelling)(sprime.s); // The labelling function always = L(s) i.e. L(s, qbar) = L(s)
                let qprime: u32 = (dfa.delta)(state.q[j], label); // this references the product function directly and is therefore always relevant
                let mut qprime_new: Vec<u32> = state.q[..j].to_vec();
                qprime_new.push(qprime);
                if sprime.s == state.s && qprime_new == state.q { self_loop = true }
                if sprime.s != state.s && qprime_new != state.q { self_that = true }
                //println!("s': {}, s: {}, s==s': {}", sprime.s,state.s, sprime.s==state.s);
                //println!("q': {:?}, q: {:?}, q==q': {}, self_loop: {}, self_that: {}", state.q, qprime_new, state.q==qprime_new, self_loop, self_that);
                println!("(s,q), a, (s',q') = p: ({},{:?}),{},({},{:?}) = {}, label: {}", state.s, state.q, transition.a, sprime.s, qprime_new, sprime.p, label);
                transition_to.push(ProductTransitionPair{s: sprime.s, p: sprime.p, q: qprime_new});
            }
            //println!("self loop: {}, that loop: {}", self_loop, self_that);
            let mut snew_cond: bool = false;
            if self_loop == true && self_that == true { snew_cond = true; }
            new_product_transitions.push(ProductTransition {
                s: state.s,
                q: state.q.to_vec(),
                a: transition.a,
                s_prime: transition_to,
                self_loop: snew_cond,
            });
        }
    }
    (new_product_states, new_product_transitions)
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

// lets just start with the action set and the state space
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