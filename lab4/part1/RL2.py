import numpy as np
import MDP
from sympy import *

class RL2:
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
        state -- current state
        action -- action to be executed

        Outputs:
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action, state])
        cumProb = np.cumsum(self.mdp.T[action, state, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def sampleSoftmaxPolicy(self, policyParams, state):
        '''从随机策略中采样单个动作的程序，通过以下概率公式采样
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        本函数将被reinforce()调用来选取动作

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs:
        action -- sampled action

        提示：计算出概率后，可以用np.random.choice()，来进行采样
        '''

        # temporary value to ensure that the code compiles until this
        # function is coded
        action = 0
        prop = np.exp(policyParams[:, state]) /  np.exp(policyParams[:, state]).sum()
        action = np.random.choice(np.arange(self.mdp.nActions), p=prop)
        return action



    def epsilonGreedyBandit(self, nIterations):
        '''Epsilon greedy 算法 for bandits (假设没有折扣因子).
        Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        # epsilon = 1 意味着只有eploration
        empiricalMeans = np.zeros(self.mdp.nActions)
        reward_list = []
        empiricalHits = np.zeros(self.mdp.nActions)
        for epoch in range(nIterations):
            epsilon = 1 / (epoch + 1)
            if epsilon < np.random.random():  # epsilon selection
                arm = np.argmax(empiricalMeans)  # exploitation
            else:
                arm = np.random.randint(0, self.mdp.nActions)  # exploration
            reward, _ = self.sampleRewardAndNextState(0, arm)
            reward_list.append(reward)
            
            # update
            empiricalMeans[arm] = (empiricalHits[arm] * empiricalMeans[arm] + reward) / (empiricalHits[arm] + 1)  # update R(a)
            empiricalHits[arm] += 1   # update na
            
        return empiricalMeans,reward_list

    def thompsonSamplingBandit(self, prior, nIterations, k=1):
        '''Thompson sampling 算法 for Bernoulli bandits (假设没有折扣因子)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards


        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)

        提示：根据beta分布的参数，可以采用np.random.beta()进行采样
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        reward_list = []
        empiricalHits = np.zeros(self.mdp.nActions)
        for _ in range(nIterations):
            # propability with beta
            prop_list = [np.random.beta(prior[arm, 0], prior[arm, 1]) for arm in range(self.mdp.nActions)]  # beta function to produce probability
            arm = np.argmax(prop_list)
            reward, _ = self.sampleRewardAndNextState(0, arm)
            reward_list.append(reward)
            
            # update
            empiricalMeans[arm] = (empiricalHits[arm] * empiricalMeans[arm] + reward) / (empiricalHits[arm] + 1)  # update R(a)
            empiricalHits[arm] += 1   # update na
            
            if reward > 0: # update a means win
                prior[arm][0] += 1
            else:
                prior[arm][1] += 1 # update b means loss
        return empiricalMeans,reward_list

    def UCBbandit(self, nIterations):
        '''Upper confidence bound 算法 for bandits (假设没有折扣因子)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        reward_list = []
        empiricalHits = np.zeros(self.mdp.nActions)
        for epoch in range(nIterations):
            prop_list = [empiricalMeans[arm] + np.sqrt((2 * np.log(epoch + 1)) / empiricalHits[arm])
                         for arm in range(self.mdp.nActions)]  # UCB propability
            arm = np.argmax(prop_list)
            reward, _ = self.sampleRewardAndNextState(0, arm)
            reward_list.append(reward)
            #update
            
            empiricalMeans[arm] = (empiricalHits[arm] * empiricalMeans[arm] + reward) / (empiricalHits[arm] + 1)  # update R(a)
            empiricalHits[arm] += 1   # update na

        return empiricalMeans,reward_list

    def reinforce(self, s0, initialPolicyParams, nEpisodes, nSteps):
        '''reinforce 算法，学习到一个随机策略，建模为：
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        上面的sampleSoftma xPolicy()实现该方法，通过调用sampleSoftmaxPolicy(policyParams,state)来选择动作
        并且同学们需要根据上课讲述的REINFORCE算法，计算梯度，根据更新公式，完成策略参数的更新。
        其中，超参数：折扣因子gamma=0.95，学习率alpha=0.01

        Inputs:
        s0 -- 初始状态
        initialPolicyParams -- 初始策略的参数 (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs:
        policyParams -- 最终策略的参数 (array of |A|x|S| entries)
        rewardList --用于记录每个episodes的累计折扣奖励 (array of |nEpisodes| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policyParams = np.zeros((self.mdp.nActions,self.mdp.nStates))
        gamma = 0.95
        alpha = 0.01
        rewardList = []
        policyParams = initialPolicyParams
        
        for epoch in range(nEpisodes):
            # generate episode
            env_state, env_action, env_reward = [s0], [], []
            grad_list = np.zeros(nSteps)
            for step in range(nSteps):
                action = self.sampleSoftmaxPolicy(policyParams, env_state[step])
                env_action.append(action)
                reward, state = self.sampleRewardAndNextState(env_state[step], env_action[step])
                env_reward.append(reward)
                env_state.append(state)
                
            # reversely get G_n
            for step in reversed(range(nSteps)):
                if step == nSteps-1:
                    grad_list[step] = env_reward[step]
                else:
                    grad_list[step] = gamma * grad_list[step+1] + env_reward[step]
                
            # gradient
            for step in range(nSteps):
                policyParams[env_action[step]][env_state[step]] += alpha * np.power(gamma, step+1) * grad_list[step]    #梯度

            rewardList.append(np.array(env_reward).sum())
            
        return [policyParams,rewardList]