use crate::mdp_structures::{ModifiedProductMDP, ProductStateSpace, TransitionPair, TaskProgress, TaskAction, TraversalStateSpace};
use std::collections::{HashMap, HashSet};
use rand::seq::SliceRandom;

#[derive(Debug, Clone)]
pub struct TeamStateSpace{
    pub r: u8,
    pub s: u32,
    pub q: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct TeamTransitionPair{
    pub r: u8,
    pub s: u32,
    pub p: f32,
    pub q: Vec<u32>,
    pub abstract_label: HashMap<u8, TaskProgress>
}

#[derive(Debug, Clone)]
pub struct TeamTransition{
    pub r: u8,
    pub s: u32,
    pub q: Vec<u32>,
    pub a: TaskAction,
    pub s_prime: Vec<TeamTransitionPair>,
    pub abstract_label: HashMap<u8, TaskProgress>,
    pub stoppable: bool,
}

impl TeamTransition {
    fn default() -> TeamTransition {
        TeamTransition {
            r: 0,
            s: 0,
            q: vec![],
            a: TaskAction { a: 0, task: 0 },
            s_prime: vec![],
            abstract_label: Default::default(),
            stoppable: false
        }
    }
}

#[derive(Debug)]
pub struct TeamTraversal {
    pub a: TaskAction,
    pub data: TeamStateSpace,
    pub p: f32,
    pub abstract_label: HashMap<u8, TaskProgress>
}

#[derive(Debug)]
pub struct TeamTraversalStateSpace {
    state: TeamStateSpace,
    a: TaskAction,
}

impl TeamTraversalStateSpace {
    fn default() -> TeamTraversalStateSpace {
        TeamTraversalStateSpace {
            state: TeamStateSpace {
                r: 0,
                s: 0,
                q: vec![]
            },
            a: TaskAction { a: 0, task: 0 }
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
                q: Vec::new()
            },
            p: 0.0,
            abstract_label: HashMap::new(),
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

/// The implementation of the team MDP
impl TeamMDP {
    pub fn empty() -> TeamMDP {
        TeamMDP {
            states: Vec::new(),
            initial: TeamStateSpace { r: 0, s: 0, q: Vec::new() },
            transitions: Vec::new(),
            robot_count: 0,
            task_count: 0,
        }
    }

    pub fn introduce_modified_mdp(& mut self, mlp: &ModifiedProductMDP) -> & mut TeamMDP {
        // create a new robot number
        self.robot_count = self.robot_count + 1;
        let team_states_new = TeamMDP::extend_to_team_product_state_space(&mlp.states, self.robot_count);
        self.states.extend(team_states_new);
        //println!("Team state space new: {:?}", self.states);
        // We will always be required to add the transitions of the modified local product MDP so we do that next
        // we cannot just copy the transitions because this will lead to non-identifiable local products

        for transition in mlp.transitions.iter() {
            let mut sprimes: Vec<TeamTransitionPair> = Vec::new();
            for sprime in transition.s_prime.iter(){
                sprimes.push(TeamTransitionPair{
                    r: self.robot_count,
                    s: sprime.s,
                    p: sprime.p,
                    q: sprime.q.to_vec(),
                    abstract_label: sprime.abstract_label.clone(),
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
                s_prime: sprimes,
                abstract_label: transition.abstract_label.clone(),
                stoppable: transition.stoppable
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
            for prev_r_transition in self.transitions.iter().filter(|x| x.r == self.robot_count - 1 && x.stoppable == true) {
                // 2. We just add a new transition which says that we are now moving to the initial state of the next automaton
                // So I guess this is what I was talking about, we have to add a switch transition to any of the transitions included in our transition vector
                // which have the properties that we are looking for, namely that r_{i-1} + 1 = r, s = 0, qbar = q'bar
                let switch_prime = TeamTransitionPair{
                    r: self.robot_count,
                    s: 0,
                    p: 1.,
                    q: prev_r_transition.q.to_vec(),
                    abstract_label: prev_r_transition.abstract_label.clone(),
                };
                switch_transitions.push(TeamTransition{
                    r: self.robot_count - 1,
                    s: prev_r_transition.s,
                    q: prev_r_transition.q.to_vec(),
                    a: TaskAction { a: 99, task: prev_r_transition.a.task }, // TODO we have hardcoded a limitation into the TaskAction struct, we will need to address this at some point
                    s_prime: vec![switch_prime],
                    abstract_label: prev_r_transition.abstract_label.clone(),
                    stoppable: prev_r_transition.stoppable
                });
            }
            self.transitions.append(&mut switch_transitions);
        } else {
            self.initial.r = self.robot_count;
            self.initial.s = mlp.initial.s;
            self.initial.q = mlp.initial.q.to_vec();
        }
        self
    }

    /// This is a helper function for the modified product MDP before it enters the team
    pub fn extend_to_team_product_state_space(mod_state_space: &Vec<ProductStateSpace>, rbot_number: u8) -> Vec<TeamStateSpace>{
        let mut team_state_space: Vec<TeamStateSpace> = Vec::new();
        let rnew: u8 = rbot_number;
        for state in mod_state_space.iter() {
            team_state_space.push(TeamStateSpace{
                r: rnew,
                s: state.s,
                q: state.q.to_vec(),
            })
        }
        team_state_space
    }

    pub fn determine_choices(&self, r: &u8, s: &u32, q: &Vec<u32>) -> TeamTraversalStateSpace {
        // for r,s,q determine the transitions that meet these criteria
        // we can do this with a filter, and use a closure to reduce the transitions to the input params
        let mut choices: Vec<TeamTransition> = Vec::new();
        for transition in self.transitions.iter().filter(|x| x.s == *s && x.r == *r && x.q == q) {
            choices.push(transition.clone());
        }
        match choices.choose(&mut rand::thread_rng()) {
            Some(x) => TeamTraversalStateSpace{
                state: TeamStateSpace {
                    r: x.r,
                    s: x.s,
                    q: x.q.to_vec()
                },
                a: TaskAction {a: x.a.a, task: x.a.task }
            },
            None => TeamTraversalStateSpace::default()
        }
    }

    pub fn team_traversal(&self){
        let mut finished: bool = false;
        let mut new_state = TeamTraversal::default();
        let tasks: Vec<u8> = (0..self.task_count).collect();

        // We should turn the following into a choice function
        println!("initial; r: {}, s:{}, q:{:?}", self.initial.r, self.initial.s, self.initial.q);
        let transition_choice  = self.determine_choices(&self.initial.r, &self.initial.s, &self.initial.q);
        let current_state = TeamTraversalStateSpace{
            state: TeamStateSpace {
                r: transition_choice.r,
                s: transition_choice.s,
                q: transition_choice.q.to_vec()
            },
            a: TaskAction{ a: initial_choice.a.a, task: initial_choice.a.task },
        };

        let mut current_state = TeamTraversalStateSpace::default();
        match initial_choices.choose(&mut rand::thread_rng()) {
            Some(initial_choice) => {

            },
            None => {  }
        }

        println!("{:?}", current_state);

        new_state = self.traversal(&current_state);
        println!("{:?}", new_state)
        //while !finished {}
    }

    fn traversal(&self, input: &TeamTraversalStateSpace) -> TeamTraversal {
        // this function is supposed to find paths through a graph
        // starting at the initial state we find a path through the product MDP
        //let a_current: i8 = 1;
        let mut output: Vec<TeamTraversal> = Vec::new();
        for x in self.transitions.iter().filter(|x| x.s == input.state.s && x.q == input.state.q && x.a.a == input.a.a && x.a.task == input.a.task) {
            println!("State found: ({},{:?},action:{:?}, (s',q'): {:?}, stoppable: {}", x.s, x.q, x.a, x.s_prime, x.stoppable);
            //println!("-> s': {:?}", x.s_prime);
            let o = x.s_prime.choose(&mut rand::thread_rng());
            match o {
                Some(traversal) => output.push(
                    TeamTraversal {
                        a: x.a.clone(),
                        data: TeamStateSpace{
                            r: traversal.r,
                            s: traversal.s,
                            q: traversal.q.to_vec()
                        },
                        p: traversal.p,
                        abstract_label: traversal.abstract_label.clone(),
                    }
                ),
                None => println!("nothing")
            }
        }
        let result = output.choose(&mut rand::thread_rng());
        match result {
            Some(x) => TeamTraversal{a: x.a, data: TeamStateSpace{ r: x.data.r, s: x.data.s, q: x.data.q.to_vec()}, p: x.p, abstract_label: x.abstract_label.clone()},
            None => {println!("filter was 0 length");TeamTraversal::default()}
        }
    }
}


