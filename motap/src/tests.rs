use crate::mdp_structures::{ModProductTransition, ProductStateSpace,
                            DFA, MDP, ProductMDP, ModifiedProductMDP};
use model_checking::{Transition, TransitionPair};
use std::convert::TryFrom;
use std::collections::HashMap;
use minilp::{Variable, LinearExpr};
use model_checking::value_iteration;

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

#[cfg(test)]
mod tests {
    use super::*;
    use minilp::Variable;
    use itertools::assert_equal;

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
        use minilp::{Problem, OptimizationDirection, ComparisonOp};

        let h = [[0.2, 0.7]];
        let k = [[0.4, 0.6], [0., 0.66]];

        const DIM: u8 = 2;
        let mut problem = Problem::new(OptimizationDirection::Maximize);

        let mut vars: HashMap<String, Variable> = HashMap::new();
        for i in 0..DIM {
            vars.insert(format!("w{}", i), problem.add_var(0., (0., 1.)));
        }
        vars.insert(format!("delta"), problem.add_var(1.0, (f64::NEG_INFINITY, f64::INFINITY)));
        let b = problem.add_var(0., (f64::NEG_INFINITY, f64::INFINITY));
        for x in h.iter() {
            let mut lhs = LinearExpr::empty();
            for j in 0..DIM {
                lhs.add(*vars.get(&*format!("w{}", j)).unwrap(), x[j as usize]);
            }
            lhs.add(b, 1.0);
            lhs.add(*vars.get("delta").unwrap(), -1.0);
            problem.add_constraint(lhs, ComparisonOp::Ge, 0.);
        }
        for x in k.iter() {
            let mut lhs = LinearExpr::empty();
            for j in 0..DIM {
                lhs.add(*vars.get(&*format!("w{}", j)).unwrap(), x[j as usize]);
            }
            lhs.add(b, 1.0);
            lhs.add(*vars.get("delta").unwrap(), 1.0);
            problem.add_constraint(lhs, ComparisonOp::Le, 0.);
        };
        let mut lhs = LinearExpr::empty();
        for j in 0..DIM {
            lhs.add(*vars.get(&*format!("w{}", j)).unwrap(), 1.0)
        }
        problem.add_constraint(lhs, ComparisonOp::Eq, 1.);

        let solution = problem.solve().unwrap();
        let w0 = (solution[*vars.get("w0").unwrap()] * 100.).round() / 100.;
        let w1 = (solution[*vars.get("w1").unwrap()] * 100.).round() / 100.0;
        assert_eq!(w0, 0.13);
        assert_eq!(w1, 0.87);
        assert_eq!(w0 + w1, 1.0);
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