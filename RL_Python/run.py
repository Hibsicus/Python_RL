# -*- coding: utf-8 -*-
from env import Maze
from RL import QLearningTable
from RL import SarsaTable
from RL import SarsaLambdaTable


def updateSarsaLambda():
    for episode in range(100):
        # initial observation
        observation = q_env.reset()

        # RL choose action based on observation
        action = rl.choose_action(str(observation))

        # initial all zero eligibility trace
        rl.eligibility_trace *= 0

        while True:
            # fresh env
            q_env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = q_env.step(action)

            # RL choose action based on next observation
            action_ = rl.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            rl.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    q_env.destroy()

def updateSarsa():
    for episode in range(100):
        
        observation = q_env.reset()
        
        action = rl.choose_action(str(observation))
        
        
        
        while True:
            q_env.render()
            
            observation_, reward, done = q_env.step(action)
            
            action_ = rl.choose_action(str(observation_))
           
            rl.learn(str(observation), action, reward, str(observation_), action_, )

            observation = observation_
            action = action_
            
            if done:
                break
            
def updateQLearn():
    for episode in range(100):
        observation = q_env.reset()
        
        while True:
            q_env.render()
            
            action = rl.choose_action(str(observation))
            
            observation_, reward, done = q_env.step(action)
            
            rl.learn(str(observation), action, reward, str(observation_))
            
            observation = observation_
            
            if done:
                break
    print('Game Over')
    q_env.destroy()


if __name__ == '__main__':
    q_env = Maze()
    rl = SarsaLambdaTable(actions=list(range(q_env.n_actions)))
    
    q_env.after(100, updateSarsaLambda)
    q_env.mainloop()