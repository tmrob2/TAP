mod mdp_structures;
mod team_mdp_structures;

use mdp_structures::{TaskProgress, Transition, TransitionPair, ProductMDP, ProductTransition, ProductStateSpace};//, product_mdp_v4};
//use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use crate::mdp_structures::{ModifiedProductMDP, ProductTransitionPair, ModProductTransition};
use itertools::{any, Itertools};
use rand::seq::SliceRandom;
use ordered_float::NotNan;
use crate::team_mdp_structures::{TeamDFSResult, TeamStateSpace, TeamMDP};
extern crate petgraph;
use petgraph::graph::Graph;
use petgraph::dot::Dot;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::panic::resume_unwind;
//extern crate ndarray;
//use ndarray::prelude::*;
//#[macro_use]
//extern crate rulinalg;

fn main() {

    let states: Vec<u32> = (0..4).collect();
    // I believe that we can actually define a non-zero rewards vector on this structure, we can
    // also use this structure to define some LTL properties on
    let transitions: Vec<Transition> = vec![
        Transition {
            s: 0,
            a: 1,
            s_prime: vec![TransitionPair{s: 1, p: 1.}],
            rewards: 0.
        },
        Transition {
            s: 1,
            a: 1,
            s_prime: vec![TransitionPair{s: 0, p: 0.6}, TransitionPair{s: 2, p: 0.3}, TransitionPair{s:3, p:0.1}],
            rewards: 0.
        },
        Transition {
            s: 1,
            a: 2,
            s_prime: vec![TransitionPair{s: 2, p: 0.5}, TransitionPair{s: 3, p: 0.5}],
            rewards: 0.
        },
        Transition {
            s: 2,
            a: 1,
            s_prime: vec![TransitionPair{s: 2, p: 1.}],
            rewards: 0.
        },
        Transition {
            s: 3,
            a: 1,
            s_prime: vec![TransitionPair { s: 3, p: 1. }],
            rewards: 0.
        }
    ];
    /*for i in transitions.iter() {
        println!("s: {}, a: {}, s': {:?}", i.s, i.a, i.s_prime.iter());
    }*/

    let transitions2: Vec<Transition> = vec![
        Transition {
            s: 0,
            a: 1,
            s_prime: vec![TransitionPair{s: 0, p: 0.2}, TransitionPair{ s:1, p:0.8}],
            rewards: 0.
        },
        Transition {
            s: 1,
            a: 1,
            s_prime: vec![TransitionPair{s:2, p:1.}],
            rewards: 0.
        },
        Transition {
            s: 2,
            a: 1,
            s_prime: vec![TransitionPair{s: 3, p: 0.5}, TransitionPair{s: 4, p: 0.5}],
            rewards: 0.
        },
        Transition {
            s: 2,
            a: 2,
            s_prime: vec![TransitionPair{s:4, p:1.}],
            rewards: 0.
        },
        Transition {
            s: 3,
            a: 1,
            s_prime: vec![TransitionPair{s:2, p:1.}],
            rewards: 0.
        },
        Transition {
            s: 4,
            a: 1,
            s_prime: vec![TransitionPair{s: 0, p: 1.}],
            rewards: 0.
        }
    ];

    let target = vec![2];
    let s0min = vec![3];
    //println!("{:?}", value_iteration(&states, &transitions, 0.001, &target, &s0min));

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

    let dfa1: mdp_structures::DFA = mdp_structures::DFA{
        states: vec![0,1,2,3],
        initial: 0u32,
        delta: delta1,
        rejected: vec![3u32],
        accepted: vec![2u32],
    };
    let dfa2 = mdp_structures::DFA {
        states: vec![0,1,2,3,4],
        initial: 0,
        delta: delta2,
        rejected: vec![4u32],
        accepted: vec![3u32],
    };
    let mdp1 = mdp_structures::MDP {
        states: vec![0,1,2,3,4],
        initial: 0,
        transitions: transitions2,
        labelling: mdp_labelling
    };
    // create an initial product MDP
    //let mut delta_hash: HashMap<u8, &fn(&u32, &str) -> u32> = HashMap::new();
    let mut empty_product_mdp: ProductMDP = ProductMDP{
        states: vec![],
        transitions: vec![],
        initial: ProductStateSpace { s: 0, q: vec![] },
        labelling: vec![mdp_labelling],
        task_counter: 0,
        dfa_delta: &mut Default::default(),
        mdp_transitions: &vec![]
    };
    let mut pmdp1 = mdp1.initial_product_mdp(&dfa1, &mut empty_product_mdp);
    //let f = **pmdp1.dfa_delta.get(&0).unwrap();
    //pmdp1.labelling = mdp_labelling2;
    let label2: L = mdp_labelling2;
    let mut pmdp2 = pmdp1.local_product(&dfa2, &1u8, &label2);

    // print the characteristics of the product mdp
    //println!("The initial state: {:?}", pmdp2.initial);
    //println!("The enumerated transitions");
    //println!("{:?}", pmdp2.states);
    /*for transition in pmdp2.transitions.iter().filter(|x| x.s == 1 && x.q == vec![0, 1]) {
        println!("{:?}", transition)
    }*/
    //pmdp2.traverse_n_steps();

    let mut base_prod_mdp = ModifiedProductMDP{
        states: vec![],
        transitions: vec![],
        initial: ProductStateSpace { s: 0, q: vec![] },
        labelling: &pmdp2.labelling,
        number: 0,
        task_counter: 0,
    };
    let mut local_prod1 = base_prod_mdp.generate_mod_product(pmdp2);
    local_prod1.assign_state_index_to_transition();
    //println!("{:?}", local_prod1.states);
    // Show an example where all of the states are unique
    /*for state in local_prod1.transitions.iter().filter(|x| x.abstract_label.values().any(|y| match y { TaskProgress::JustFailed => true, _ => false})) {
        println!("{:?}", state);
    }*/
    //local_prod1.traverse_n_steps();
    let local_prod2 = local_prod1.clone();
    let mut t = team_mdp_structures::TeamMDP::empty();
    let num_agents: u8 = 2;
    t.introduce_modified_mdp(&local_prod1, &num_agents);
    println!("Team mdp state space index test: {}", t.check_transition_index());
    /*for transition in t.transitions.iter() {
        println!("{:?}", transition)
    }*/
    // TODO we need to test traversal in the MDP structure with more than one team member
    t.introduce_modified_mdp(&local_prod2, &num_agents);
    /*for transition in t.transitions.iter().filter(|x| match x.abstract_label.get(&x.a.task) {
        Some(y) => match y {
            TaskProgress::JustFailed => true,
            _ => false
        },
        None => false
    } == true && x.a.a != 99) {
        println!("{:?}", transition)
    }*/
    /*for transition in t.transitions.iter().filter(|x| x.r == 2 && x.s == 0) {
        println!("{:?}", transition);
    }*/
    //t.team_traversal();
    let w: Vec<f32> = vec![0.,0.,0.,1.];
    //t.minimise_expected_weighted_cost_of_scheduler(&w)
    /// Show that for every transition there is exactly one state representing this transition, and
    /// for every state there are a non-empty set of transition which represent that state
    let local_prod_sound: bool = show_state_space_is_transition_space(&local_prod1.transitions, &local_prod1.states);
    assert_eq!(local_prod_sound, true); // this says that every state in the state space is
    let statespace_index_test: bool = local_prod1.check_transition_index();
    // represented by a transition in the list of transitions
    println!("Mod prod state space index test: {}", statespace_index_test);
    println!("Team mdp state space index test: {}", t.check_transition_index());
    //t.minimise_expected_weighted_cost_of_scheduler(&w);
    // Show all of the illegal states -> we want to demonstrate that these states are unreachable
    // from the initial state of the agent, and should thus be discounted
    let mut illegal_states: HashSet<TeamStateSpace> = HashSet::new();
    for transition in t.transitions.iter().filter(|x| x.abstract_label.values().all(|x| match x { TaskProgress::InProgress => true, _ => false} == true)){
        illegal_states.insert(TeamStateSpace{
            r: transition.r,
            s: transition.s,
            q: transition.q.to_vec(),
            switch_to: false,
            stoppable: transition.stoppable,
            active_task: if transition.stoppable { 99 } else { match TeamMDP::active_task(&transition.abstract_label) {Some(y) => y, None => 99 }}
        });
    }
    println!("# States: {}", t.states.len());
    println!("# transitions: {}", t.transitions.len());
    //let result: (TeamBFSResult, String) = t.bfs();
    let result: TeamDFSResult = t.reachable_states();

    //let mut excluded: HashSet<TeamStateSpace> = result.not_visited.into_iter().collect();
    //println!("{:?}", illegal_states.difference(&excluded));
    println!("# visited states: {}", result.visted.len());

    // Code which generates a graph of the team MDP
    /*
    let g: String = t.generate_graph(&result.visted);
    let mut file = File::create("graph.dot").unwrap();
    file.write_all(&g.as_bytes());
     */

    // Running algorithm 2 in pieces
    /*let mu = t.minimise_expected_weighted_cost_of_scheduler(&result.visted, &w, 0.001);
    let g_mu = t.construct_scheduler_graph(&mu);
    let mut file = File::create("graph_mu1_2.dot").unwrap();
    file.write_all(&g_mu.as_bytes());
    /*for tran in t.transitions.iter().filter(|x| x.r == 2 && x.s == 2 && x.q == vec![3,0]){
        println!("({},{},{:?}), action: {:?} -> {:?}", tran.r, tran.s, tran.q, tran.a, tran.s_prime);
    }*/

     */
}

fn delta1<'a>(q: u32, a: &'a str) -> u32 {
    match (q, a) {
        (0, a) => if ["initiate1"].contains(&a) { 1} else { 0},
        (1, a) => if ["ready", "initiate1", "none"].contains(&a) { 1 } else if ["sprint1"].contains(&a) { 2 } else { 3 },
        (2, a) => 2,
        (3, a) => 3, // this should throw an error, therefore we will have to handle it
        _ => q,
    }
}

fn delta2<'a>(q: u32, a: &'a str) -> u32 {
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

type L<'a> = fn(s: u32) -> &'a str;

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

// tests

fn show_state_space_is_transition_space(transitions: &Vec<ModProductTransition>, state_space: &Vec<ProductStateSpace>) -> bool {
    for state in state_space.iter() {
        let mut filt = transitions.iter().filter(|x| x.s == state.s && x.q == state.q).peekable();
        if filt.peek().is_none(){
            println!("{}, {:?}", state.s, state.q);
            return false
        }
    }
    true
}