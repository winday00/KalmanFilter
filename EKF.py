#扩展卡尔曼滤波是卡尔曼滤波在非线性系统上的扩展，利用泰勒函数的一阶展开式将非线性系统转换为线性系统，再利用标准卡尔曼的方法进行求解
#EKF的转移方程和观测方式如下：
#
# X(k+1) = f(X(k)) + Qs
# Z(k+1) = h(X(k+1)) + Vs
# 其中f和h为非线性函数
#下面以一个实际应用的场景对EKF进行说明
#假设物体移动的状态向量为S= [X,Y,Vx,Vy],观测参数为Z = (R,that),利用扩展卡尔曼滤波对移动轨迹就行修正估计

import numpy as np
import matplotlib.pyplot as plt
import math

class EKF(object):
    def __init__(self):
        self.initial_state = np.array([[1000], [1500], [5], [-3]])
        self.i_iter = 100
        # 初始化转移噪声和观测噪声协方差
        self.VarianceQs = np.array([[2], [2], [0.2], [0.2]])
        self.VarianceRs = np.array([[10], [0.001]])
        #根据过程协方差得到过程噪声和观测噪声
        self.ProcessNoise = self.VarianceQs.repeat(self.i_iter, axis=1) * np.random.randn(len(self.VarianceQs), self.i_iter)
        self.ObservateNoise = self.VarianceRs.repeat(self.i_iter, axis=1) * np.random.randn(len(self.VarianceRs), self.i_iter)
        self.Qs = np.cov(self.ProcessNoise)
        self.Rs = np.cov(self.ObservateNoise)
        #构造状态转移矩阵
        self.A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        #初始化协方差矩阵
        self.variance = np.eye(len(self.initial_state))


    def hthta(self, state):
        result = np.zeros((len(self.VarianceRs), 1))
        for i in range(state.shape[1]):
            x = state[0,i]
            y = state[1,i]
            vx = state[2,i]
            vy = state[3,i]
            r = np.sqrt(x*x+y*y)
            thta = math.atan(y/x)
            result = np.hstack((result,np.array([[r],[thta]])))
        result = result[:,1:state.shape[1]+1]
        return result

    def MatrixH(self,state):
        #构造观测函数的雅克比矩阵，本实例中观测函数为rk = sqrt(Xk^2 + Yk^2) thtak = arctan(yk/xk),为非线性方程
        #构造雅克比矩阵如下：https://blog.csdn.net/weixin_42647783/article/details/89054641
        x = float(state[0])
        y = float(state[1])
        vx = float(state[2])
        vy = float(state[3])
        H00 = x / np.sqrt(x*x+y*y)
        H01 = y / np.sqrt(x*x+y*y)
        H02 = 0
        H03 = 0
        H10 = -y/(x*x+y*y)
        H11 = x/(x*x+y*y)
        H12 = 0
        H13 = 0
        H = np.array([[H00, H01, H02, H03], [H10, H11, H12, H13]])
        return H

    def Predict(self, State, Variance):
        #根据当前状态预测下一时刻状态和协方差
        State_P = np.dot(self.A, State)
        variance_P = np.dot(np.dot(self.A, Variance), self.A.T) + self.Qs
        return State_P, variance_P

    def Update(self, State_P, variance_P, Zk):#预测状态、预测方差、当前时刻观测值
        #得到观测方程的雅克比矩阵
        H = self.MatrixH(State_P)
        #计算kalman增益
        Sk = np.linalg.inv(np.dot(np.dot(H, variance_P), H.T) + self.Rs)
        kg = np.dot(np.dot(variance_P, H.T), Sk)
        #计算预测值的观测结果
        zh = self.hthta(State_P)
        #更新预测值
        State_U = State_P + np.dot(kg,(Zk - zh))
        variance_U = np.dot(np.eye(len(self.initial_state)) - np.dot(kg, H), variance_P)
        return State_U, variance_U

    def State2Observation(self,Observation):
        result = np.zeros((len(self.VarianceRs), 1))
        for i in range(Observation.shape[1]):
            R = Observation[0,i]
            tanthta = Observation[1,i]
            X = R * math.cos(tanthta)
            Y = R * math.sin(tanthta)
            result = np.hstack((result, np.array([[X], [Y]])))
        result = result[:, 1:Observation.shape[1] + 1]
        return result

    def run(self):
        IdealState = self.initial_state
        #生成理论数据值
        for i in range(1, self.i_iter):
            nextState = np.dot(self.A, IdealState[:, i - 1]).reshape(len(self.initial_state),1)
            IdealState = np.hstack((IdealState, nextState))
        RealState = IdealState + self.ProcessNoise
        ObervationState = self.hthta(RealState) + self.ObservateNoise
        KalmanState = self.initial_state
        for k in range(1, self.i_iter):
            KnownState = self.initial_state
            State_P,Variance_P = self.Predict(KnownState, self.variance)
            State_U, Variance_U = self.Update(State_P, Variance_P, ObervationState[:, k].reshape(len(ObervationState[:, k]), 1))
            KalmanState = np.hstack((KalmanState, State_U))
            self.variance = Variance_U
            self.initial_state = State_U
        ObservationValue = self.State2Observation(ObervationState)
        #结果绘图
        IdealX = IdealState[0, :]
        IdealY = IdealState[1, :]
        ObserX = ObservationValue[0, :]
        ObserY = ObservationValue[1, :]
        KalmanX = KalmanState[0, :]
        KalmanY = KalmanState[1, :]
        plt.plot(IdealX, IdealY, color='y', ls="-", lw=1, label="Ideal")
        plt.plot(ObserX, ObserY, c='r', ls="-", lw=1, label="Observation")
        plt.plot(KalmanX, KalmanY, c='b', ls="-", lw=1, label="Kalman")
        plt.legend()
        plt.show()





if __name__=='__main__':
    testEKF = EKF()
    testEKF.run()









































