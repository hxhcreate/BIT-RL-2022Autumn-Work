import math

import numpy as np
import MDP


class RL:
    def __init__(self, mdp, sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self, state, action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
zhenate
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        # 按照当前R值为均值根据高斯密度函数得到随机奖励
        reward = self.sampleReward(self.mdp.R[action, state])
        cumProb = np.cumsum(self.mdp.T[action, state, :])  # 把array当成一维数组不断向右累加
        nextState = np.where(cumProb >= np.random.rand(1))[
            0][0]  # 返回第一个满足条件的state
        return [reward, nextState]

    def qLearning(self, s0, initialQ, nEpisodes, nSteps, epsilon=0, temperature=0):
        '''
        qLearning算法，需要将Epsilon exploration和 Boltzmann exploration 相结合。
        以epsilon的概率随机取一个动作，否则采用 Boltzmann exploration取动作。
        当epsilon和temperature都为0时，将不进行探索。

        Inputs:
        s0 -- 初始状态
        initialQ -- 初始化Q函数 (|A|x|S| array)
        nEpisodes -- 回合（episodes）的数量 (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- 每个回合的步数(steps)
        epsilon -- 随机选取一个动作的概率
        temperature -- 调节 Boltzmann exploration 的参数

        Outputs: 
        Q -- 最终的 Q函数 (|A|x|S| array)
        policy -- 最终的策略
        rewardList -- 每个episode的累计奖励（|nEpisodes| array）
        '''
        rewardList = []  # 保存每一轮的reward
        Q = initialQ
        nCounts = np.zeros_like(Q)  # 初始化一个被访问的计数array 方便计算learning_rate
        for episode in range(nEpisodes):
            rewardEpisode = 0
            curState = s0
            for step in range(nSteps):
                if (np.random.rand(1) <= epsilon):  # 随机选取
                    selectAction = np.random.randint(self.mdp.nActions)
                elif temperature != 0:   # boltzmann选取
                    probActionBoltzman = np.exp(
                        Q[:, curState] / temperature) / np.sum(np.exp(Q[:, curState] / temperature))
                    selectAction = np.where(
                        np.cumsum(probActionBoltzman) >= np.random.rand(1))[0][0]
                else:  # 特殊判断如果温度是0的话，我们采取绝对的贪婪策略，不进行任何的探索
                    selectAction = np.argmax(Q[:, curState])
                [reward, nextState] = self.sampleRewardAndNextState(  # 观测下一个状态的奖励
                    curState, selectAction)
                rewardEpisode += reward
                nCounts[selectAction, curState] += 1
                lr = 1 / nCounts[selectAction, curState]  # 学习率是访问该点次数的倒数
                Q[selectAction, curState] += lr * (reward + self.mdp.discount * Q[:, nextState].max() - Q[selectAction, curState])  # 更新Q
                curState = nextState  # 执行状态转移
            rewardList.append(rewardEpisode)

        policy = self.mdp.extractPolicy(Q.max(axis=0))  # 利用mdp中已实现的函数直接提取policy
        return [Q, policy, rewardList]
