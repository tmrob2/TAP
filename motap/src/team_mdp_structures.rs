use crate::mdp_structures::{ModifiedProductMDP, ProductStateSpace, TransitionPair, TaskProgress, TaskAction, TraversalStateSpace, tasks_finished, is_stoppable};
use std::collections::{HashMap, HashSet, VecDeque};
use rand::seq::SliceRandom;
use std::hash::Hash;
use itertools::assert_equal;
use ordered_float::NotNan;
use ndarray::arr1;
extern crate petgraph;
use petgraph::graph::Graph;
use petgraph::dot::Dot;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;

pub struct TeamDFSResult {
    pub visted: Vec<TeamStateSpace>,
    pub not_visited: Vec<TeamStateSpace>
}

#[derive(Debug, Clone, Hash, Eq, PartialEq,)]
pub struct TeamStateSpace {
    pub r: u8,
    pub s: u32,
    pub q: Vec<u32>,
    pub switch_to: bool,
    pub stoppable: bool,
    pub active_task: u8,
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
                active_task: 99
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
                active_task: 99
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
    for (i,x) in u.iter().enumerate() {
        sum_value += u[i] * v[i]
    }
    sum_value
}

/// The implementation of the team MDP
impl TeamMDP {
    pub fn empty() -> TeamMDP {
        TeamMDP {
            states: Vec::new(),
            initial: TeamStateSpace { r: 0, s: 0, q: Vec::new(), switch_to: false, stoppable: false, active_task: 99},
            transitions: Vec::new(),
            robot_count: 0,
            task_count: 0,
        }
    }

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

        // todo we need to write a function to make sure that every state in the team state space corresponds to the correct transition,
        //  so that we can be sure that when we are iterating in the algorithm that we are using the correct states, actions and probabilities

        for transition in mlp.transitions.iter() {
            let mut sprimes: Vec<TeamTransitionPair> = Vec::new();
            let mut rewards_model_values: Vec<f32> = vec![0.; *agent_capacity as usize];
            let mut task_reward_values: Vec<f32> = vec![0.; self.task_count as usize];

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
                    active_task: 99,
                });
                switch_transitions.push(TeamTransition{
                    r: self.robot_count - 1,
                    s: prev_r_transition.s,
                    q: prev_r_transition.q.to_vec(),
                    rewards_model: vec![0.; (self.task_count + agent_capacity) as usize],
                    a: TaskAction { a: 99, task: prev_r_transition.a.task }, // TODO we have hardcoded a limitation into the TaskAction struct, we will need to address this at some point
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
            self.initial.active_task = 99;
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
                        active_task: 99
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
                        active_task: 99
                    });
                }
            }
        }
        team_state_space
    }

    pub fn determine_choices(&self, r: &u8, s: &u32, q: &Vec<u32>) -> TeamTraversalStateSpace {
        // for r,s,q determine the transitions that meet these criteria
        // we can do this with a filter, and use a closure to reduce the transitions to the input params
        let mut choices: Vec<TeamTransition> = Vec::new();
        for transition in self.transitions.iter().filter(|x| x.s == *s && x.r == *r && x.q == *q) {
            choices.push(transition.clone());
        }
        match choices.choose(&mut rand::thread_rng()) {
            Some(x) => TeamTraversalStateSpace{
                state: TeamStateSpace {
                    r: x.r,
                    s: x.s,
                    q: x.q.to_vec(),
                    switch_to: false,
                    stoppable: x.stoppable,
                    active_task: 99
                },
                a: TaskAction {a: x.a.a, task: x.a.task },
                abstract_label: x.abstract_label.clone(),
                stoppable: x.stoppable
            },
            None => TeamTraversalStateSpace::default()
        }
    }

    fn label(&self, r: &u8, s: &u32, q: &Vec<u32>, a: &TaskAction) -> Vec<TaskProgress> {
        let mut ap_return: Vec<TaskProgress> = Vec::new();
        for transition in self.transitions.iter().filter(|x| x.s == *s && x.q == *q && x.a.a == a.a && x.a.task == a.task && x.r == *r){
            ap_return.extend(transition.abstract_label.values().cloned().collect::<Vec<TaskProgress>>());
        }
        //let ap_return_hash: HashSet<_> = ap_return.iter().cloned().collect();
        //let ap_return_unique: Vec<String> = ap_return_hash.into_iter().collect();
        //ap_return_unique
        ap_return
    }

    pub fn team_traversal(&self) {
        let mut finished: bool = false;
        let mut new_state = TeamTraversal::default();

        // We should turn the following into a choice function
        println!("initial; r: {}, s:{}, q:{:?}", self.initial.r, self.initial.s, self.initial.q);
        let mut transition_choice  = self.determine_choices(&self.initial.r, &self.initial.s, &self.initial.q);
        let mut current_state = TeamTraversalStateSpace {
            state: TeamStateSpace {
                r: transition_choice.state.r,
                s: transition_choice.state.s,
                q: transition_choice.state.q.to_vec(),
                switch_to: transition_choice.state.switch_to,
                stoppable: transition_choice.stoppable,
                active_task: 99
            },
            a: TaskAction{ a: transition_choice.a.a, task: transition_choice.a.task },
            abstract_label: transition_choice.abstract_label.clone(),
            stoppable: transition_choice.stoppable,
        };

        println!("{:?}", current_state);
        while !finished {
            // it is important that the new state and the current state both contain
            new_state = self.traversal(&current_state);
            print!("p((r: {}, s:{},q{:?}) , a:{:?}, (r':{}, s':{},q':{:?}))={}: ", &current_state.state.r, &current_state.state.s, &current_state.state.q, &new_state.a, &new_state.data.r, &new_state.data.s, &new_state.data.q, &new_state.p);
            println!("abstract label: {:?} -> {:?}", current_state.abstract_label, new_state.abstract_label);
            current_state = TeamTraversalStateSpace{ state: TeamStateSpace {
                r: new_state.data.r,
                s: new_state.data.s,
                q: new_state.data.q.to_vec(),
                switch_to: new_state.data.switch_to,
                stoppable: new_state.stoppable,
                active_task: 99
            },
                a: TaskAction { a: new_state.a.a, task: new_state.a.task },
                abstract_label: new_state.abstract_label.clone(),
                stoppable: new_state.stoppable
            };
            // If the current state is stoppable then we need to consider two things
            // 1. If the current state is stoppable then we are required to choose a new action to
            // proceed to.
            if current_state.stoppable {
                if tasks_finished(&current_state.abstract_label) {
                    finished = true;
                }
                else {
                    transition_choice = self.determine_choices(&current_state.state.r, &current_state.state.s, &current_state.state.q);
                    let mut new_choice_state = TeamTraversalStateSpace {
                        state: TeamStateSpace {
                            r: transition_choice.state.r,
                            s: transition_choice.state.s,
                            q: transition_choice.state.q.to_vec(),
                            switch_to: transition_choice.state.switch_to,
                            stoppable: transition_choice.stoppable,
                            active_task: 99
                        },
                        a: TaskAction { a: transition_choice.a.a, task: transition_choice.a.task },
                        abstract_label: transition_choice.abstract_label.clone(),
                        stoppable: transition_choice.stoppable
                    };
                    // 2. have all tasks been completed, how can we measure if all tasks have been completed
                    // The easiest way to do this is to just check whether all tasks are in a state of fail,
                    // finished, or justFail
                    println!("Picking a new action");
                    print!("p((r: {}, s:{},q{:?}) , a:{:?}, (r':{}, s':{},q':{:?}))={}: ", &current_state.state.r, &current_state.state.s, &current_state.state.q, &new_state.a, &new_state.data.r, &new_state.data.s, &new_state.data.q, &new_state.p);
                    println!("abstract label: {:?} -> {:?}", current_state.abstract_label, new_state.abstract_label);
                    current_state = new_choice_state
                }
            }
        }
    }

    fn traversal(&self, input: &TeamTraversalStateSpace) -> TeamTraversal {
        // this function is supposed to find paths through a graph
        // starting at the initial state we find a path through the product MDP
        //let a_current: i8 = 1;
        let mut output: Vec<TeamTraversal> = Vec::new();
        for x in self.transitions.iter().filter(|x| x.s == input.state.s && x.q == input.state.q && x.a.a == input.a.a && x.a.task == input.a.task) {
            //println!("State found: ({},{:?},action:{:?}, (s',q'): {:?}, stoppable: {}", x.s, x.q, x.a, x.s_prime, x.stoppable);
            //println!("-> s': {:?}", x.s_prime);
            let o = x.s_prime.choose(&mut rand::thread_rng());
            match o {
                Some(traversal) => output.push(
                    TeamTraversal {
                        a: x.a.clone(),
                        data: TeamStateSpace{
                            r: traversal.r,
                            s: traversal.s,
                            q: traversal.q.to_vec(),
                            switch_to: false,
                            stoppable: traversal.stoppable,
                            active_task: 99
                        },
                        p: traversal.p,
                        abstract_label: traversal.abstract_label.clone(),
                        stoppable: traversal.stoppable,
                    }
                ),
                None => println!("nothing")
            }
        }
        let result = output.choose(&mut rand::thread_rng());
        match result {
            Some(x) => TeamTraversal{
                a: x.a,
                data: TeamStateSpace{
                    r: x.data.r,
                    s: x.data.s,
                    q: x.data.q.to_vec(),
                    switch_to: x.data.switch_to,
                    stoppable: x.stoppable,
                    active_task: 99
                },
                p: x.p, abstract_label: x.abstract_label.clone(), stoppable: x.stoppable},
            None => {println!("filter was 0 length");TeamTraversal::default()}
        }
    }

    fn inner_stoppable_newtask<'a>(&self, action_values: &'a mut Vec<f32>, actions: &'a mut Vec<TaskAction>, state: &TeamStateSpace, xbar: &Vec<f32>, w: &Vec<f32>) -> () {
        for transition in self.transitions.iter().filter(|x| x.s == state.s && x.r == state.r && x.q == state.q && x.a.task != state.active_task) {
            //println!("old state: {:?}, new transition: {:?}", state, transition.s_prime);
            let mut sprime_values_a: Vec<f32> = Vec::new();
            //println!("rewards: {:?}", transition.rewards_model);
            for sprime in transition.s_prime.iter() {
                //println!("a: {:?}, s': {:?}, vect index: {}", transition.a, sprime, k);
                sprime_values_a.push(sprime.p * xbar[sprime.state_index]);
            }
            let sum_value: f32 = sprime_values_a.iter().sum();
            let new_value: f32 = norm(w, &transition.rewards_model) + sum_value;
            //println!("normal: {}", norm(w, &transition.rewards_model) );
            action_values.push(new_value);
            actions.push(transition.a);
        }
    }

    fn inner_nonstoppable_tasks<'a>(&self, action_values: &'a mut Vec<f32>, actions: &'a mut Vec<TaskAction>, state: &TeamStateSpace, xbar: &Vec<f32>, w: &Vec<f32>) -> () {
        for transition in self.transitions.iter().filter(|x| x.s == state.s && x.r == state.r && x.q == state.q && x.a.task == state.active_task) {
            let mut sprime_values_a: Vec<f32> = Vec::new();
            for sprime in transition.s_prime.iter() {
                //println!("a: {:?}, s': {:?}, vect index: {}", transition.a, sprime, k);
                sprime_values_a.push(sprime.p * xbar[sprime.state_index]);
            }
            let sum_value: f32 = sprime_values_a.iter().sum();
            let new_value: f32 = norm(w, &transition.rewards_model) + sum_value;
            action_values.push(new_value);
            actions.push(transition.a);
        }
    }

    /// The inner loop function is the standard inner operation of the value iteration, only the state
    /// space changes which depends on the agent being considered.
    fn inner_loop_actions(&self, state: &TeamStateSpace, xbar: &mut Vec<f32>, w: &Vec<f32>, k: &usize) -> (f32, TaskAction) {
        // TODO how do I check whether the state is the correct state or not
        let mut action_values: Vec<f32> = Vec::new();
        let mut actions: Vec<TaskAction> = Vec::new();
        // Show the set of legal actions, these are the actions
        // two courses 1: there all actions are stoppable
        //             2: there exists a state which is not stoppable
        //println!("State assessed: {:?}", state);
        if state.stoppable {
            // There are more actions available if the state is stoppable as opposed to when it is not stoppable
            //println!("state: {:?}", state);
            // if the state is the initial state of the MPD, and it is stoppable for the current task, then we need to
            // choose an action from the next set of tasks which does not include the current task..
            self.inner_stoppable_newtask(&mut action_values, &mut actions, &state, &xbar, &w);
        } else {
            self.inner_nonstoppable_tasks(&mut action_values, &mut actions, &state, &xbar, &w);
        }

        let non_nan_floats: Vec<_> = action_values.iter().cloned().map(NotNan::new).filter_map(Result::ok).collect();
        let min = non_nan_floats.iter().min().unwrap();
        let index = non_nan_floats.iter().position(|x| x == min).unwrap();
        let size = non_nan_floats.len();
        let minf32: f32 = min.into_inner();
        //println!("min: {}, index: {}, size: {}, values: {:?}", minf32, index, size, non_nan_floats);
        (minf32, actions[index])
    }

    pub fn minimise_expected_weighted_cost_of_scheduler(&self, reachable_states: &Vec<TeamStateSpace>, w: &Vec<f32>, epsilon_0: f32) -> Vec<(TeamStateSpace, TaskAction)> {
        // initialise the action vector
        let mut mu: Vec<(TeamStateSpace, TaskAction)> = vec![(TeamStateSpace{
            r: 0,
            s: 0,
            q: vec![],
            switch_to: false,
            stoppable: false,
            active_task: 0
        }, TaskAction::default()); reachable_states.len()];
        let m = self.robot_count;
        println!("robot capacity: {}", m);
        let mut epsilon: f32 = 1.;
        //for i in (1..m).rev() {
        let i = 2;
            // initialise a bunch of vectors
        let mut xbar: Vec<f32> = vec![0.; reachable_states.len()];
        let mut ybar: Vec<f32> = vec![0.; reachable_states.len()];
        let mut X: Vec<Vec<f32>> = vec![vec![0.; reachable_states.len()]; self.robot_count as usize];
        let mut Y: Vec<Vec<f32>> = vec![vec![0.; reachable_states.len()]; self.robot_count as usize];

        while epsilon > epsilon_0 {
            if i < m {
                for (k,state) in reachable_states.iter().filter(|x| x.r == i).enumerate(){
                    let (ynew, opt_action) = self.inner_loop_actions(state, &mut xbar,  w, &k);
                    ybar[k] = ynew;
                    mu[k] = (state.clone(), opt_action);
                }
            } else {
                for (k,state) in reachable_states.iter().filter(|x| x.r == i || (x.r == i + 1 && x.switch_to)).enumerate(){
                    let (ynew, opt_action) = self.inner_loop_actions(state, &mut xbar, w, &k);
                    ybar[k] = ynew;
                    mu[k] = (state.clone(), opt_action);
                }
            }
            let xbar_arr1 = arr1(&xbar);
            let ybar_arr1 = arr1(&ybar);
            let diff = &ybar_arr1 - &xbar_arr1;
            let non_nan_eps: Vec<_> = diff.iter().cloned().map(NotNan::new).filter_map(Result::ok).collect();
            let epsilon_new = non_nan_eps.iter().max().unwrap().into_inner();
            println!("max epsilon: {}", epsilon_new);
            xbar = ybar.to_vec();
            epsilon = epsilon_new
        }
        /*for (state, action) in mu.iter() {
            if action.a >= 0 {
                println!("state: ({},{},{:?}), action: {:?}", state.r, state.s, state.q, action)
            }
        }*/
        mu
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
        //let mut graph: Graph<String, String> = Graph::new();

        queue.push_front(self.initial.clone());
        let position_init = self.states.iter().position(|x| x.r == self.initial.r && x.s == self.initial.s && x.q == self.initial.q).unwrap();
        let mut visited: Vec<bool> = vec![false; self.states.len()];
        visited[position_init] = true;
        //graph.add_node(format!("({},{},{:?})", self.initial.r, self.initial.s, self.initial.q));

        while !queue.is_empty() {
            let next_state = queue.pop_front().unwrap();
            //println!("{:?}", next_state);
            if next_state.stoppable {
                for transition in self.transitions.iter().filter(|x| x.r == next_state.r && x.s == next_state.s && x.q == next_state.q) {
                    for sprime in transition.s_prime.iter() {
                        for state_edit in self.states.iter_mut().filter(|x| x.r == sprime.r && x.s == sprime.s && x.q == sprime.q) {
                            state_edit.active_task == transition.a.task;
                        }
                        let sprime_index = self.states.iter().position(|z| z.r == sprime.r && z.s == sprime.s && z.q == sprime.q).unwrap();
                        if !visited[sprime_index] {
                            visited[sprime_index] = true;
                            //println!("Connection from: {} -> {}, with {}, index: {}", format!("({},{},{:?})", next_state.r, next_state.s, next_state.q), format!("({},{},{:?})", sprime.r, sprime.s, sprime.q), format!("({},{})", transition.a.a, transition.a.task), sprime_index);
                            let stoppable_value = is_stoppable(&sprime.abstract_label);
                            let sprime_position = self.states.iter().position(|x| x.r == sprime.r && x.s == sprime.s && x.q == sprime.q).unwrap();
                            let switch_to_value: bool = self.states[sprime_position].switch_to;
                            queue.push_front(TeamStateSpace {
                                r: sprime.r,
                                s: sprime.s,
                                q: sprime.q.to_vec(),
                                switch_to: switch_to_value,
                                stoppable: stoppable_value,
                                active_task: transition.a.task
                            });
                        }
                    }
                }
            } else {
                for transition in self.transitions.iter().filter(|x| x.r == next_state.r && x.s == next_state.s && x.q == next_state.q && x.a.task == next_state.active_task) {
                    for sprime in transition.s_prime.iter() {
                        let sprime_index = self.states.iter().position(|z| z.r == sprime.r && z.s == sprime.s && z.q == sprime.q).unwrap();
                        if !visited[sprime_index] {
                            visited[sprime_index] = true;
                            let stoppable_value = is_stoppable(&sprime.abstract_label);
                            let sprime_position = self.states.iter().position(|x| x.r == sprime.r && x.s == sprime.s && x.q == sprime.q).unwrap();
                            let switch_to_value: bool = self.states[sprime_position].switch_to;
                            //println!("Connection from: {} -> {}, with {}, index: {}", format!("({},{},{:?})", next_state.r, next_state.s, next_state.q), format!("({},{},{:?})", sprime.r, sprime.s, sprime.q), format!("({},{})", transition.a.a, transition.a.task), sprime_index);
                            queue.push_front(TeamStateSpace {
                                r: sprime.r,
                                s: sprime.s,
                                q: sprime.q.to_vec(),
                                switch_to: switch_to_value,
                                stoppable: stoppable_value,
                                active_task: 99
                            });
                        }
                    }
                }
            }
        }

        for (i,x) in visited.iter().enumerate() {
            if *x {
                visited_states.push(self.states[i].clone());
            } else {
                dead_states.push(self.states[i].clone());
            }
        }
        TeamDFSResult {
            visted: visited_states,
            not_visited: dead_states,
        }
    }

    /// Generate a graph on the reachable states of the team MDP, useful for
    /// debugging the team MDP structure
    pub fn generate_graph(&self, visited_states: &Vec<TeamStateSpace>) -> String {
        let mut graph: Graph<String, String> = Graph::new();
        let mut node_added: Vec<bool> = vec![false; visited_states.len()];
        for state in visited_states.iter() {
            let origin_index = visited_states.iter().position(|x| x.r == state.r && x.s == state.s && x.q == state.q).unwrap();
            if !node_added[origin_index] {
                graph.add_node(format!("({},{},{:?})", state.r, state.s, state.q));
                node_added[origin_index] = true;
            }
            if state.stoppable {
                // any transition is possible
                for trans in self.transitions.iter().filter(|x| x.r == state.r && x.s == state.s && x.q == state.q) {
                    for sprime in trans.s_prime.iter(){
                        // The sprime are the desitinations of the node with the transition that we search from some state,
                        // but there may be multiple transitions because we have multiple actions
                        let destination_index = visited_states.iter().position(|x| x.r == sprime.r && x.s == sprime.s && x.q == sprime.q).unwrap();
                        if !node_added[destination_index] {
                            graph.add_node(format!("({},{},{:?})", sprime.r, sprime.s, sprime.q));
                            node_added[destination_index] = true;
                        }
                        let action = format!("a: {}, task: {}", trans.a.a, trans.a.task);
                        let origin_node_index = graph.node_indices().find(|i| graph[*i] == format!("({},{},{:?})", state.r, state.s, state.q)).unwrap();
                        let destination_node_index = graph.node_indices().find(|i| graph[*i] == format!("({},{},{:?})", sprime.r, sprime.s, sprime.q)).unwrap();
                        graph.add_edge(origin_node_index, destination_node_index, action);
                    }
                }
            } else {
                // we need to know what the active task is an filter the transitions to actions relating to that task
                //println!("state: {:?}", state);
                for trans in self.transitions.iter().filter(|x| x.r == state.r && x.s == state.s && x.q == state.q && x.a.task == state.active_task) {
                    for sprime in trans.s_prime.iter(){
                        // The sprime are the desitinations of the node with the transition that we search from some state,
                        // but there may be multiple transitions because we have multiple actions
                        let destination_index = visited_states.iter().position(|x| x.r == sprime.r && x.s == sprime.s && x.q == sprime.q).unwrap();
                        if !node_added[destination_index] {
                            graph.add_node(format!("({},{},{:?})", sprime.r, sprime.s, sprime.q));
                            node_added[destination_index] = true;
                        }
                        let action = format!("a: {}, task: {}", trans.a.a, trans.a.task);
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
                    let sprime_position = match mu.iter().position(|(x,y)| x.s == sprime.s && x.r == sprime.r && x.q == sprime.q) {
                        Some(x) => x,
                        None => {println!("The prime is not contained within the scheduler"); 0}
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


