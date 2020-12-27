use crate::mdp_structures::{ModifiedProductMDP, TaskProgress, TaskAction};
use std::collections::{HashMap, HashSet, VecDeque};
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

pub struct TeamDFSResult {
    pub visted: Vec<TeamStateSpace>,
    pub not_visited: Vec<TeamStateSpace>
}

struct StateActionPair {
    state: TeamStateSpace,
    action_set: Vec<TaskAction>
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
    pub p: f32,
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
    pub rewards_model: Vec<f32>,
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
    pub p: f32,
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
pub fn norm(u: &Vec<f32>, v: &Vec<f32>) -> f32 {
    assert_eq!(u.len(), v.len());
    let mut sum_value: f32 = 0.;
    for (i,_x) in u.iter().enumerate() {
        sum_value += u[i] * v[i]
    }
    sum_value
}

/// The implementation of the team MDP
impl TeamMDP {
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
            let mut rewards_model_values: Vec<f32> = vec![0.; *agent_capacity as usize];
            let mut task_reward_values: Vec<f32> = vec![0.; self.task_count as usize];
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
                let mut switch_rewards: Vec<f32> = vec![0.; (*agent_capacity + self.task_count) as usize];
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
    /// This is done through the use of the state_index attribute, which represents the location of the state in the
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
    fn inner_action_optimisation(&self, state: &TeamStateSpace, xbar: &Vec<f32>, w_arr1: &Array1<f32>, reachable_states: &Vec<TeamStateSpace>, verbose: &bool) -> (f32, TaskAction) {
        let mut action_values: Vec<f32> = Vec::new();
        let mut actions: Vec<TaskAction> = Vec::new();
        if *verbose {
            println!("state: ({},{},{:?})", state.r, state.s, state.q);
        }
        for action in state.action_set.iter(){
            for transition in self.transitions.iter().
                filter(|x| x.r == state.r && x.s == state.s && x.q == state.q && x.a.a == action.a && x.a.task == action.task) {
                let rewards = arr1(&transition.rewards_model);
                let norm = rewards.dot(w_arr1);
                let mut sum_sprime_values: Vec::<f32> = Vec::new();
                for sprime in transition.s_prime.iter(){
                    let sprime_position = reachable_states.iter().position(|x| x.r == sprime.r && x.s == sprime.s && x.q == sprime.q).unwrap();
                    sum_sprime_values.push(sprime.p * xbar[sprime_position]);
                    if *verbose {
                        println!("sprime: ({},{},{:?}), p: {}, xbar: {}:", sprime.r, sprime.s, sprime.q, sprime.p, xbar[sprime_position]);
                    }
                }
                let summed_values: f32 = sum_sprime_values.iter().sum();
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
        let ynew: f32 = min_value.into_inner();
        let opt_action: TaskAction = actions[index];
        (ynew, opt_action)
    }

    /// This private function is the second loop calculation of algorithm (2) which calculates the
    /// actual reward for each objective in the multi-objective problem for a given scheduler
    fn inner_optimal_reward_optimisation(&self, mu: &mut Vec<(TeamStateSpace, TaskAction)>, state: &TeamStateSpace,
                                         X: &mut Vec<Vec<f32>>, Y: &mut Vec<Vec<f32>>, w: &Vec<f32>, reachable_states: &Vec<TeamStateSpace>, k: &usize) -> () {
        let optimal_action = mu[*k].1;
        for transition in self.transitions.iter().
            filter(|x| x.r == state.r && x.s == state.s && x.q == state.q && x.a.a == optimal_action.a && x.a.task == optimal_action.task) {
            for j in 0..w.len() {
                let mut sum_values: Vec<f32> = Vec::new();
                for sprime in transition.s_prime.iter() {
                    let sprime_position: usize = reachable_states.iter().position(|x| x.r == sprime.r && x.s == sprime.s && x.q == sprime.q).unwrap();
                    sum_values.push(sprime.p * X[j][sprime_position])
                }
                let sum_value: f32 = sum_values.iter().sum();
                Y[j][*k] = transition.rewards_model[j] + sum_value;
            }
        }
    }

    /// This is algorithm 2, of the MOTAP theory, see paper for more details, and formalisations
    pub fn minimise_expected_weighted_cost_of_scheduler(&self, reachable_states: &Vec<TeamStateSpace>, w: &Vec<f32>, epsilon_0: f32) -> (Vec<(TeamStateSpace, TaskAction)>, Vec<f32>) {
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
        let mut r: Vec<f32> = vec![0.; w.len()];
        let w_arr1 = arr1(w);
        let m = self.robot_count;
        println!("robot capacity: {}", m);
        let mut epsilon: f32 = 1.;
        //for i in (1..m).rev() {
            // initialise a bunch of vectors
        for i in (1..2+1).rev() {
            //let i: u8 = 1;
            println!("generating the scheduler for robot: {}", i);
            let mut xbar: Vec<f32> = vec![0.; reachable_states.len()];
            let mut X: Vec<Vec<f32>> = vec![vec![0.; reachable_states.len()]; w.len()];
            let mut ybar: Vec<f32> = vec![0.; reachable_states.len()];
            let mut Y: Vec<Vec<f32>> = vec![vec![0.; reachable_states.len()]; w.len()];
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
                let mut eps_j_values: Vec<f32> = vec![1.; w.len()];

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
                    print!("y[{}] = {}, ", j, r[j]);
                }
            }
            println!("");
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


