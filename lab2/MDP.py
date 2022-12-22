from glob import escape
from re import S
from tkinter import NS
import numpy as np

class MDP:
    '''一个简单的MDP类，它包含如下成员'''

    def __init__(self,T,R,discount):
        '''构建MDP类

        输入:
        T -- 转移函数: |A| x |S| x |S'| array
        R -- 奖励函数: |A| x |S| array
        discount -- 折扣因子: scalar in [0,1)

        构造函数检验输入是否有效，并在MDP对象中构建相应变量'''

        assert T.ndim == 3, "转移函数无效，应该有3个维度"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "无效的转换函数：它具有维度 " + repr(T.shape) + ", 但它应该是(nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "无效的转移函数：某些转移概率不等于1"
        self.T = T
        assert R.ndim == 2, "奖励功能无效：应该有2个维度"
        assert R.shape == (self.nActions,self.nStates), "奖励函数无效：它具有维度 " + repr(R.shape) + ", 但它应该是 (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "折扣系数无效：它应该在[0,1]中"
        self.discount = discount
        
    # 没用到    
    def one_state_best_action(self, state, V):
        value_q = np.zeros(self.nActions)
        for action in range(self.nActions):
            for next_state in range(self.nStates):
                prob = self.T[action, state, next_state]
                reward = self.R[action, state]
                value_q[action] += prob *(reward + self.discount * V[next_state])
        return value_q
    
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''值迭代法
        V <-- max_a R^a + gamma T^a V

        输入: 
        initialV -- 初始的值函数: 大小为|S|的array
        nIterations -- 迭代次数的限制：标量 (默认值: infinity)
        tolerance -- ||V^n-V^n+1||_inf的阈值: 标量 (默认值: 0.01)

        Outputs: 
        V -- 值函数: 大小为|S|的array
        iterId -- 执行的迭代次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        iterId = 0
        epsilon = 0
        V = initialV
        #填空部分
        # 循环做法，不够good
        # while iterId < nIterations:
        #     epsilon = 0 
        #     iterId += 1
        #     for state in range(self.nStates):
        #         best_action_value = np.max(self.one_state_best_action(state, V))
        #         epsilon = max(epsilon, np.abs(best_action_value - V[state]))
        #         V[state] = best_action_value
        #     if epsilon < tolerance :
        #         break
        
        # 矩阵做法
        # while iterId < nIterations:
        #     iterId += 1
        #     prev_v = np.copy(V)
        #     q_sa = np.zeros((self.nActions, self.nStates))
        #     for next_state in range(self.nStates):
        #         q_sa += (self.T[:, next_state, :] * V)
        #     best_values = self.R + self.discount * q_sa
        #     V = np.max(best_values, axis=0)
        #     epsilon = np.abs(prev_v - V).max()
        #     if epsilon < tolerance:
        #         break
        
        # 还可以再化简
        while iterId < nIterations:
            iterId += 1
            prev_v = np.copy(V)
            best_values = self.R + self.discount * (self.T * V).sum(axis=2)  # self.T * V 这里有一个广播机制，结果还是a * s * s, 将next_state 累加 得到 a * s 的矩阵
            best_value = np.max(best_values, axis=0)  # 在nAaction维度做sum 可以得到最好的actino对应的V
            epsilon = np.abs(prev_v - best_value).max()  # 取最大的差值和tolerance做比较
            V = best_value  # 赋值
            if epsilon < tolerance:
                break
        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''从值函数中提取具体策略的程序
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- 值函数: 大小为|S|的array

        Output:
        policy -- 策略: 大小为|S|的array'''
        #填空部分
        policy = np.zeros(V.shape, dtype=np.int)
        best_values = self.R + self.discount * (self.T * V).sum(axis=2)
        policy = np.argmax(best_values, axis=0)
        return policy.astype(np.int) 

    def evaluatePolicy(self,policy):
        '''通过求解线性方程组来评估政策
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- 策略: 大小为|S|的array

        Ouput:
        V -- 值函数: 大小为|S|的array'''
        #填空部分
        V = np.zeros(self.nStates) 
        # for state in range(self.nStates):
        #     V = np.matmul(np.linalg.inv(np.eye(self.nStates) - self.discount * self.T[policy[state], :, :]), 
        #                          self.R[policy[state], :].reshape(self.nStates, -1)) 
        Tpai = np.array([self.T[policy[state], state] for state in range(self.nStates)])
        Rpai = np.array([self.R[policy[state], state] for state in range(self.nStates)])
        return np.matmul(np.linalg.inv(np.eye(self.nStates)-self.discount*Tpai), Rpai).flatten()  # 线性代数
    
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''策略迭代程序:  在策略评估(solve V^pi = R^pi + gamma T^pi V^pi) 和
        策略改进 (pi <-- argmax_a R^a + gamma T^a V^pi)之间多次迭代

        Inputs:
        initialPolicy -- 初始策略: 大小为|S|的array
        nIterations -- 迭代数量的限制: 标量 (默认值: inf)

        Outputs: 
        policy -- 策略: 大小为|S|的array
        V -- 值函数: 大小为|S|的array
        iterId --策略迭代执行的次数 : 标量'''


        V = np.zeros(self.nStates)
        policy = initialPolicy
        iterId = 0
        V = np.zeros(self.nStates)
        while iterId < nIterations:
            iterId += 1
            V = self.evaluatePolicy(policy)  # 先策略评估再策略提升
            policy_st = self.extractPolicy(V)
            # unchanged = True
            # for state in range(self.nStates):
            #     rewards = (self.T[:, state, :] * V).sum(axis=1)
            #     now_action = policy[state]
            #     if rewards.max() > (self.T[now_action, state, :] * V).sum():
            #         policy[state] = np.argmax(rewards)
            #         unchanged = False
            # if unchanged:
            #     break
            if (policy == policy_st).all():  # 需要全部相等 取all
                break
            else:
                policy = policy_st
                
        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''部分的策略评估:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- 策略: 大小为|S|的array
        initialV -- 初始的值函数: 大小为|S|的array
        nIterations -- 迭代数量的限制: 标量 (默认值: infinity)
        tolerance --  ||V^n-V^n+1||_inf的阈值: 标量 (默认值: 0.01)

        Outputs: 
        V -- 值函数: 大小为|S|的array
        iterId -- 迭代执行的次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        V = initialV
        iterId = 0
        epsilon = 0
        #填空部分
        Tpai = np.array([self.T[policy[state], state] for state in range(self.nStates)])
        Rpai = np.array([self.R[policy[state], state] for state in range(self.nStates)])

        while iterId < nIterations:
            iterId += 1
            prev_v = np.copy(V)
            V = Rpai + self.discount * (Tpai * V).sum(axis=1)
            epsilon = np.max(np.abs(prev_v - V))
            if epsilon < tolerance:
                break
        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''修改的策略迭代程序: 在部分策略评估 (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        和策略改进(pi <-- argmax_a R^a + gamma T^a V^pi)之间多次迭代

        Inputs:
        initialPolicy -- 初始策略: 大小为|S|的array
        initialV -- 初始的值函数: 大小为|S|的array
        nEvalIterations -- 每次部分策略评估时迭代次数的限制: 标量 (默认值: 5)
        nIterations -- 修改的策略迭代中迭代次数的限制: 标量 (默认值: inf)
        tolerance -- ||V^n-V^n+1||_inf的阈值: scalar (默认值: 0.01)

        Outputs: 
        policy -- 策略: 大小为|S|的array
        V --值函数: 大小为|S|的array
        iterId -- 修改后策略迭代执行的迭代次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        iterId = 0
        epsilon = 0
        policy = initialPolicy
        V = initialV
        #填空部分
        policy = initialPolicy
        iterId = 0
        while iterId <= nIterations:
            iterId += 1
            V,_, epsilon = self.evaluatePolicyPartially(policy, V, nEvalIterations, tolerance)
            policy = self.extractPolicy(V)
            if epsilon < tolerance:
                break
            
        return [policy,V,iterId,epsilon]
        