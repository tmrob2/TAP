
use model_checking::{value_iteration, generate_random_vector_sum1, member_closure_set, pareto_lp, witness, muliobj_scheduler_synthesis};

use model_checking::{Transition, TransitionPair, TeamDFSResult,
                     TeamStateSpace, TaskProgress, ProductMDP, ProductStateSpace,
                     DFA, MDP, ModifiedProductMDP, ModProductTransition,
                     TeamMDP};
//use itertools::Itertools;
use std::collections::{HashSet, HashMap};
use std::convert::TryFrom;
//use itertools::{any, Itertools};
//use rand::seq::SliceRandom;
//use ordered_float::NotNan;;//, TeamMDP, norm};
//extern crate petgraph;
//use petgraph::graph::Graph;
//use petgraph::dot::Dot;
//use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use ndarray::arr1;

fn main() {

    let transitions2: Vec<Transition> = vec![
        Transition {
            s: 0,
            a: 1,
            s_prime: vec![TransitionPair{s: 0, p: 0.1}, TransitionPair{ s:1, p:0.9}],
            rewards: 1.
        },
        Transition {
            s: 1,
            a: 1,
            s_prime: vec![TransitionPair{s:2, p:1.}],
            rewards: 1.
        },
        Transition {
            s: 2,
            a: 1,
            s_prime: vec![TransitionPair{s: 3, p: 0.9}, TransitionPair{s: 4, p: 0.1}],
            rewards: 1.
        },
        Transition {
            s: 2,
            a: 2,
            s_prime: vec![TransitionPair{s:4, p:1.}],
            rewards: 1.
        },
        Transition {
            s: 3,
            a: 1,
            s_prime: vec![TransitionPair{s:2, p:1.}],
            rewards: 1.
        },
        Transition {
            s: 4,
            a: 1,
            s_prime: vec![TransitionPair{s: 0, p: 1.}],
            rewards: 1.
        }
    ];

    //let j_task: u32 = 0;

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
    //let mut delta_hash: HashMap<u8, &fn(&u32, &str) -> u32> = HashMap::new();
    let mut empty_product_mdp: model_checking::ProductMDP = model_checking::ProductMDP{
        states: vec![],
        transitions: vec![],
        initial: model_checking::ProductStateSpace { s: 0, q: vec![], mdp_init: false },
        labelling: vec![mdp_labelling],
        task_counter: 0,
        dfa_delta: &mut Default::default(),
        mdp_transitions: &vec![]
    };
    let pmdp1 = mdp1.initial_product_mdp(&dfa1, &mut empty_product_mdp);
    //let f = **pmdp1.dfa_delta.get(&0).unwrap();
    //pmdp1.labelling = mdp_labelling2;
    let label2: L = mdp_labelling2;
    let pmdp2 = pmdp1.local_product(&dfa2, &1u8, &label2);

    // print the characteristics of the product mdp
    //println!("The initial state: {:?}", pmdp2.initial);
    //println!("The enumerated transitions");
    //println!("{:?}", pmdp2.states);
    /*for transition in pmdp2.transitions.iter().filter(|x| x.s == 1 && x.q == vec![0, 1]) {
        println!("{:?}", transition)
    }*/
    //pmdp2.traverse_n_steps();

    let mut base_prod_mdp = model_checking::ModifiedProductMDP{
        states: vec![],
        transitions: vec![],
        initial: model_checking::ProductStateSpace { s: 0, q: vec![], mdp_init: false},
        labelling: &pmdp2.labelling,
        number: 0,
        task_counter: 0,
    };
    let local_prod1 = base_prod_mdp.generate_mod_product(pmdp2);
    local_prod1.assign_state_index_to_transition();
    //println!("{:?}", local_prod1.states);
    // Show an example where all of the states are unique
    /*for state in local_prod1.transitions.iter().filter(|x| x.abstract_label.values().any(|y| match y { TaskProgress::JustFailed => true, _ => false})) {
        println!("{:?}", state);
    }*/
    //local_prod1.traverse_n_steps();
    let local_prod2 = local_prod1.clone();
    let mut t = TeamMDP::empty();
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

    let statespace_index_test: bool = local_prod1.check_transition_index();
    // represented by a transition in the list of transitions
    println!("Mod prod state space index test: {}", statespace_index_test);
    println!("Team mdp state space index test: {}", t.check_transition_index());
    //t.minimise_expected_weighted_cost_of_scheduler(&w);
    // Show all of the illegal states -> we want to demonstrate that these states are unreachable
    // from the initial state of the agent, and should thus be discounted
    // todo move the illegal states to the test section
    let mut illegal_states: HashSet<model_checking::TeamStateSpace> = HashSet::new();
    for transition in t.transitions.iter().filter(|x| x.abstract_label.values().all(|x| match x { model_checking::TaskProgress::InProgress => true, _ => false} == true)){
        illegal_states.insert(model_checking::TeamStateSpace{
            r: transition.r,
            s: transition.s,
            q: transition.q.to_vec(),
            switch_to: false,
            stoppable: transition.stoppable,
            action_set: Vec::new(),
            mdp_init: false
        });
    }
    println!("# States: {}", t.states.len());
    println!("# transitions: {}", t.transitions.len());

    let result: model_checking::TeamDFSResult = t.reachable_states();

    //let mut excluded: HashSet<TeamStateSpace> = result.not_visited.into_iter().collect();
    //println!("{:?}", illegal_states.difference(&excluded));
    println!("# visited states: {}", result.visted.len());
    /*for state in result.visted.iter() {
        for t in t.transitions.iter().
            filter(|x| x.r == state.r && x.s == state.s && x.q == state.q && x.a.a == 99) {
            println!("state: ({},{},{:?}), rewards: {:?}, action: {:?}, label: {:?}", t.r, t.s, t.q, t.rewards_model, t.a, t.abstract_label)
        }
    }*/

    /*for state in result.visted.iter() {
        println!("visited state: ({},{},{:?}), actions available: {:?}", state.r, state.s, state.q, state.action_set)
    }*/
    // Code which generates a graph of the team MDP and also generates the state action pairs which
    // will be used to minimise the expected total cost of the schedulers
    /*let g: String = t.generate_graph(&mut result.visted);
    let mut file = File::create("graph_new.dot").unwrap();
    file.write_all(&g.as_bytes());
    */

    // Running algorithm 2 in pieces
    //let w: Vec<f64> = vec![0.0,0.5,0.0,0.5];
    let target = vec![12.,12.,0.5,0.5];
    //let (mu, r) = t.minimise_expected_weighted_cost_of_scheduler(&result.visted, &w, 0.001);
    //println!("output norm: {}", arr1(&w).dot(&arr1(&r)));
    //println!("target norm: {}", arr1(&w).dot(&arr1(&target)));

    /*
    let g_mu = t.construct_scheduler_graph(&mu);
    let mut file = File::create("graph_mu1_1.dot").unwrap();
    file.write_all(&g_mu.as_bytes());

     */

    let sched_output = muliobj_scheduler_synthesis(&t, &target, &result.visted);

}

fn show_state_space_is_transition_space(transitions: &Vec<model_checking::ModProductTransition>, state_space: &Vec<ProductStateSpace>) -> bool {
    for state in state_space.iter() {
        let mut filt = transitions.iter().filter(|x| x.s == state.s && x.q == state.q).peekable();
        if filt.peek().is_none(){
            println!("{}, {:?}", state.s, state.q);
            return false
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use minilp::Variable;
    use itertools::assert_equal;
    use model_checking::member_closure_set;

    #[test]
    /// Test that an empty hashmap produces a lenght of zero
    fn test_hmap_empty() {
        let hmap: HashMap<u32, String> = HashMap::new();
        assert_eq!(hmap.len(), 0);
    }

    #[test]
    /// Test that basic value iteration is working on a simple MDP
    fn test_value_iteration() -> Result<(), String>{
        let target = vec![2];
        let s0min = vec![3];
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
        value_iteration(&states, &transitions, 0.001, &target, &s0min);
        Ok(())
    }

    #[test]
    /// Show that for every transition there is exactly one state representing this transition, and
    /// for every state there are a non-empty set of transition which represent that state
    fn state_space_transitions(){
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
        //let mut delta_hash: HashMap<u8, &fn(&u32, &str) -> u32> = HashMap::new();
        let mut empty_product_mdp: ProductMDP = ProductMDP{
            states: vec![],
            transitions: vec![],
            initial: ProductStateSpace { s: 0, q: vec![], mdp_init: false },
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

        let mut base_prod_mdp = ModifiedProductMDP{
            states: vec![],
            transitions: vec![],
            initial: ProductStateSpace { s: 0, q: vec![], mdp_init: false},
            labelling: &pmdp2.labelling,
            number: 0,
            task_counter: 0,
        };
        let mut local_prod1 = base_prod_mdp.generate_mod_product(pmdp2);
        let local_prod_sound: bool = show_state_space_is_transition_space(&local_prod1.transitions, &local_prod1.states);
        assert_eq!(local_prod_sound, true); // this says that every state in the state space is
    }

    #[test]
    fn milp_test() {
        use minilp::{Problem, OptimizationDirection, ComparisonOp};

        // Maximize an objective function x + 2 * y of two variables x >= 0 and 0 <= y <= 3
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_var(1.0, (0.0, f64::INFINITY));
        let y = problem.add_var(2.0, (0.0, 3.0));

        // subject to constraints: x + y <= 4 and 2 * x + y >= 2.
        problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint(&[(x, 2.0), (y, 1.0)], ComparisonOp::Ge, 2.0);

        // Optimal value is 7, achieved at x = 1 and y = 3.
        let solution = problem.solve().unwrap();
        assert_eq!(solution.objective(), 7.0);
        assert_eq!(solution[x], 1.0);
        assert_eq!(solution[y], 3.0);
    }

    #[test]
    fn motap_milp_problem_formulation() {

        let h: Vec<Vec<f64>> = vec![vec![0.2, 0.7]];
        let k: Vec<Vec<f64>> = vec![vec![0.4, 0.6], vec![0., 0.66]];

        let w = model_checking::pareto_lp(&h, &k, &2);
        //assert_eq!(w[0], 0.13);
        //assert_!(w[1], 0.87);
        assert_eq!(w[0] + &w[1], 1.0);
    }

    #[test]
    fn test_random_val_vec() {
        let output = model_checking::generate_random_vector_sum1(&4,&0, &100);
        let sum_solution: f64 = ((output.iter().fold(0., |sum, val| sum + val) * 1000.).round() / 1000.) as f64;
        assert_eq!(sum_solution, 1.0);
    }

    #[test]
    fn test_closure_set() {
        let hull_set_t = vec![vec![0.2, 0.1], vec![0.3, 0.4], vec![0.6, 0.5]];
        let hull_set_f = vec![vec![0.3, 0.7], vec![0.5, 0.2]];
        let target = vec![0.4, 0.5];
        assert_eq!(member_closure_set(&hull_set_t, &target), true);
        assert_eq!(member_closure_set(&hull_set_f, &target), false);
    }
}

pub fn delta1<'a>(q: u32, a: &'a str) -> u32 {
    match (q, a) {
        (0, a) => if ["initiate1"].contains(&a) { 1} else { 0},
        (1, a) => if ["ready", "initiate1", "none"].contains(&a) { 1 } else if ["sprint1"].contains(&a) { 2 } else { 3 },
        (2, _a) => 2,
        (3, _a) => 3, // this should throw an error, therefore we will have to handle it
        _ => q,
    }
}

pub fn delta2<'a>(q: u32, a: &'a str) -> u32 {
    match (q, a) {
        (0, a) => if ["initiate2"].contains(&a) {1} else {0},
        (1, a) => if ["ready", "initiate2", "none"].contains(&a) {1} else if ["sprint2"].contains(&a) {2} else {4},
        (2, a) => if ["ready", "initiate2", "none"].contains(&a) {2} else if ["sprint2"].contains(&a) {3} else {4},
        (3, _a) => 3,
        (4, _a) => 4,
        _ => q, //
    }
}

fn delta3(q:u32, a: &str) -> u32 {
    match (q, a) {
        (0, a) => if ["initiate2"].contains(&a) {1} else {0},
        (1, a) => if ["ready", "initiate2", "none"].contains(&a) {1} else if ["sprint2"].contains(&a) {2} else {5},
        (2, a) => if ["ready", "initiate2", "none"].contains(&a) {2} else if ["sprint2"].contains(&a) {3} else {5},
        (3, a) => if ["ready", "initiate2", "none"].contains(&a) {3} else if ["sprint2"].contains(&a) {4} else {5},
        (4, _a) => 4,
        (5, _a) => 5,
        _ => q, //
    }
}

pub fn mdp_labelling<'a>(s: u32) -> &'a str {
    match s {
        0 => "none",
        1 => "initiate1",
        2 => "ready",
        3 => "sprint1",
        4 => "exit",
        _ => "error",
    }
}

pub fn mdp_labelling2<'a>(s: u32) -> &'a str {
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

