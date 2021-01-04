use criterion::{black_box, criterion_group, criterion_main, Criterion};
use model_checking::*;
use std::collections::{HashMap,HashSet};
use criterion::measurement::WallTime;

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

pub fn minimisation_benchmark(c: &mut Criterion) -> &mut Criterion<WallTime> {
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
    let mut w = vec![0.25, 0.25, 0.25, 0.25];

    c.bench_function("scheduler minimisation",|b| b.iter(|| t.minimise_expected_weighted_cost_of_scheduler(&result.visted, &w, 0.001)))
}

pub fn sched_synth_benchmark(c: &mut Criterion) -> &mut Criterion<WallTime> {
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
    let target = vec![12.,12.,0.5,0.5];

    c.bench_function("scheduler minimisation",|b| b.iter(|| muliobj_scheduler_synthesis(&t, &target, &result.visted)))
}

criterion_group!(benches, minimisation_benchmark, sched_synth_benchmark);
criterion_main!(benches);


