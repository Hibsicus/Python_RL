# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time

class ObjRL(object):
    def __init__(self, action_space, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                    pd.Series(
                            [0]*len(self.actions),
                            index=self.q_table.columns,
                            name=state,
                            )
                    )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if np.random.rand() < self.epsilon:
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)
            
        return action
    
    def learn(self, *args):
        pass

class QLearningTable(ObjRL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update

            

class SarsaTable(ObjRL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, a_]
        else:
            q_target = r
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

class SarsaLambdaTable(ObjRL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay = 0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()
        
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                    )
            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)
    
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, a_]
        else:
            q_target = r
        error = q_target - q_predict
        
        #method 1
#        self.eligibility_trace.ix[s, a] += 1
        
        #method2
        self.eligibility_trace.ix[s, :] *= 0
        self.eligibility_trace.ix[s, a] = 1
        
        self.q_table += self.lr * error * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_

np.random.seed(2)

N_STATE = 6 #length of thie dimensional world
ACTIONS = ['left', 'right'] #actions
EPSILON = 0.9
ALPHA = 0.1 #learning rate
LAMBDA = 0.9 #discount
MAX_EISODES = 13 #13round
FRESH_TIME = 0.3 #time for one move

def build_q_table(n_states, actions):
    table = pd.DataFrame(
            np.zeros((n_states, len(actions))),
            columns=actions,
            )
#    print(table)
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state , :]
#    print(state_actions)
    
    if (np.random.uniform() > EPSILON) or (state_actions.all()) == 0:
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name
    
def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATE - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R
                
def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATE-1) + ['T']  #------T
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' %(episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                              ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
        
def rl():
    q_table = build_q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
                
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            
            update_env(S, episode, step_counter + 1)
            
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
    