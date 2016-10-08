import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import itertools

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha=0.9, epsilon=0.01, gamma=0.1, init_q=2):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        #   Initiate q learning vales
        self.alpha = alpha           # learning rate
        self.epsilon = epsilon      # exploration bound - default is no exploration at eps = 0
        self.gamma = gamma          # decay rate
        self.init_q = init_q        # inital q value for unexplored
        
        # Need variables to track total number of penalties and turns        
        self.penalties = 0.0
        self.updates = 0.0
        
#        with open("logs.txt", 'wb') as logs:
#            logs.write = ('Alpha: {}'.format(self.alpha))
#            logs.write('Gamma: {}'.format(self.gamma))
#            logs.write('Epsilon: {}'.format(self.epsilon))
#            logs.write('Q Initialization: {}'.format(self.init_q))
            
        # TODO: Initialize any additional variables here
        self.reward = 0.0 # cumulative reward for a trial
        
        # Initiate a Q table to hold values only when creating the agent
        # sample space is the state space of 384 variables
        # the order of the tuple is light, oncoming, left, right, next_waypoint  
        
        ss = tuple(itertools.product(['red', 'green'], self.env.valid_actions, \
        self.env.valid_actions, self.env.valid_actions, ['forward', 'left', 'right']))
        #keys = tuple(itertools.product(ss, self.env.valid_actions))
        # q values are the values of the q dict

        ### q is really q-hat until the learner converges        
        q = {}
        for s in ss:
            q[s] = {}
            for act in self.env.valid_actions:
                q[s][act] = self.init_q
        self.q = q
       
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # Destination and starting location are reset in environment.py

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Select action according to your policy
        # action = random.choice(self.env.valid_actions) # Random choice of action
                
        ### print(action)
        
        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        # print(self.state)
        
        # Printing q before update for diagnostics
#        print(self.q)
        
        ### Policy         
        ### possible random choice if epsilon non-zero
        if random.random() < self.epsilon:
            action = random.choice(self.env.valid_actions)
        ### Max action for current state in q table
        else:
            action = max(self.q[self.state], key=self.q[self.state].get)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.reward += reward
        if reward < 0:
            self.penalties += 1.0
        self.updates += 1.0
        
        # TODO: Learn policy based on state, action, reward
        # Need both state_p and action_p
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        # Pulling in s prime
        self.state_p = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        action_p = max(self.q[self.state_p], key=self.q[self.state_p].get)
        self.q[self.state][action] += self.alpha * (reward + self.gamma * self.q[self.state_p][action_p] - self.q[self.state][action])
        ### print('Learning Agent location: ', self.env.agent_states[self]['location'])
        
#        print("LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward))  # [debug]
        
        ### print q after update
#        print(self.q)        

        print('Total reward: ', self.reward)
        print('Percentage of penalized actions: ', (self.penalties/self.updates))
        #print('Total turns so far: ', self.updates)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

if __name__ == '__main__':
    run()
