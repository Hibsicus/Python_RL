# -*- coding: utf-8 -*-
from env import Maze
from QLearning import QLearningTable

def update():
    for episode in range(100):
        observation = q_env.reset()
        
        while True:
            q_env.render()
            
            action = RL.choose_action(str(observation))
            
            observation_, reward, done = q_env.step(action)
            
            RL.learn(str(observation), action, reward, str(observation_))
            
            observation = observation_
            
            if done:
                break
            

if __name__ == '__main__':
    q_env = Maze()
    RL = QLearningTable(actions=list(range(q_env.n_actions)))
    
    q_env.after(100, update)
    q_env.mainloop()