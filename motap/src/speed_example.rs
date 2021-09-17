// ------------------------------------------------------
// Model built for speed with a compile time known model
// ------------------------------------------------------
mod model_checking;
use model_checking::s_mdp::{Agent};
use model_checking::s_automaton::{GenericTask};
use model_checking::product_mdp::{ProductAgent, ProductAgentInner, Analysis};
use model_checking::s_scpm::{SCPM, ValueIteration};
use std::error::Error;
use ndarray::{Array2, arr2, arr1, s};

// Define a statically sized matrix type to handle the transition matrix for an
// MDP Agent. A static matrix is stored on the stack.
const AGENT1_SIZE: usize = 5;
const ACT: usize = 2;
const SCPM_ACT: usize = ACT;
const ALPHA_SIZE: usize = 5;
const TASK1_SIZE: usize = 4;
const TASK2_SIZE: usize = 5;
const SIZE1: usize = AGENT1_SIZE * TASK1_SIZE;
const SIZE2: usize = AGENT1_SIZE * TASK2_SIZE;
const SCPM_SIZE: usize = SIZE1 + SIZE2; // + SIZE1 + ;
const NUM_TASKS: usize = 2;
const NUM_AGENTS: usize = 1;
const N: usize = NUM_TASKS + NUM_AGENTS;

type Word<'a> = &'a str;
type StateActions = Vec<usize>;

fn main() -> Result<(), Box<dyn Error>> {
    // Set the DFA alphabet, which corresponds to the language of the MDP also
    let dfa_alphabet = [("init", 0), ("exit", 1), ("send", 2), ("ready", 3), ("start", 4)];

    // If the MDP is small then we can construct the entire transition matrix
    // Otherwise we can construct a sparse matrix and set values using a closure
    let t_a: Array2<f32> = arr2(
        &[[0.1, 0.9, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.9, 0.1],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0]]
    );
    let t_b: Array2<f32> = arr2(
        &[[0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]]
    );
    let rewards1: Array2<f32> = arr2(
        &[[1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.0, 0.0],
        [1.0, 0.0]]
    );

    let mdp_state_labels: [(usize, &str); AGENT1_SIZE] = [(0, "start"), (1, "init"), (2, "ready"), (3, "send"), (4, "exit")];
    let mdp_actions: [(usize,  StateActions); 5] = [(0, vec![0]), (1, vec![0]), (2, vec![0,1]), (3, vec![0]), (4, vec![0])];
    let actions: [&str; ACT] = ["a", "b"];
    let matrix = [&t_a, &t_b];
    let agent1: Agent<f32, Word, ACT, AGENT1_SIZE> = Agent::new(0, &actions, &mdp_actions[..], &matrix[..], &mdp_state_labels[..], &rewards1);

    let task1: GenericTask<Word, TASK1_SIZE> = GenericTask::new("task1.csv", 0, &dfa_alphabet[..], vec![2])?;
    let task2: GenericTask<Word, TASK2_SIZE> = GenericTask::new("task2.csv", 0, &dfa_alphabet[..], vec![3])?;

    //let task1agent1: ProductAgent<f32, &str, ACT, SIZE1> = ProductAgent::new(&agent1, &task1, 0,0)?;
    //let task2agent1: ProductAgent<f32, &str, ACT, SIZE2> = ProductAgent::new(&agent1, &task2, 0, 1)?;
    ProductAgent::<f32, &str, ACT, SIZE1>::new(&agent1, &task1, 0,0)?;

    // SCPM - Model
    //let mut scpm: SCPM<f32, &str, SCPM_SIZE, SCPM_ACT, N> = SCPM::default();
    //let pmdps: Vec<Box<dyn ProductAgentInner<f32, &str>>> = vec![Box::new(task1agent1), Box::new(task2agent1)]; //,
    //scpm.new(&pmdps[..], NUM_TASKS, NUM_AGENTS, "sw");
    //let target: [f32; 3] = [-5.0, 0.5, 0.5];
    //scpm.run( &target[..], &0.0005, NUM_AGENTS, NUM_TASKS, &pmdps);

    Ok(())
}

