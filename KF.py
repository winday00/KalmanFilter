#卡尔曼滤波的5个公式
# 1、预测方程：
# X(k/k-1) = A*X(k-1)+B*U(k)   K时刻X的预测状态为k-1时刻X的状态X(k-1)*A(转移矩阵)+B*U(k)，U(k)时刻的控制量
# P(k/k-1) = A*P(k-1)*A' + Q(s) k时刻的X的预测协方差为k-1时刻的协方差P(k-1)*A+ Q(s)  Q(s):系统过程的协方差，由控制量的不准确造成
# 2、更新矩阵(在获得k时刻的预测值之后，根据预测值和此时的测量值，可以对k时刻的系统状态进行更新)：
# X(k/k) = X(k/k-1) + kg(k)*(Z(k) - H*X(k/k-1)) kg为卡尔曼增益，Z(k)为k时刻的测量值(测量值和状态值并不是完全相等的，因此 另Z(k) = H*X(k)+V(s))
# 卡尔曼增益的计算方法是： kg(k) = P(k/k-1)*H' / (H* P(k/k-1)*H' + R(s) ) R(s):表示观测过程的协方差，由观测量的不准确造成
# 方差sigma的更新函数：  P(k/k) = P(k/k-1)*(I - kg(k) * H)

#恒定加速度模型
import numpy as np
import matplotlib.pyplot as plt




class KF(object):

    def __init__(self):
        #定义初始状态参数
        ax = 0.1
        ay = 0.15
        az = 0.2
        vx0 = 0
        vy0 = 0
        vz0 = 0
        x = 0
        y = 0
        z = 0
        self.initial = np.array([[x], [y], [z], [vx0], [vy0], [vz0], [ax], [ay], [az]])
        self.i_iter = 300
        #定义状态转移矩阵
        self.A = np.array([[1, 0, 0, 1, 0, 0, 0.5, 0, 0],
                          [0, 1, 0, 0, 1, 0, 0, 0.5, 0],
                          [0, 0, 1, 0, 0, 1, 0, 0, 0.5],
                          [0, 0, 0, 1, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        #定义观测

IdealState = np.vstack((Xx, Yy, Zz, Vx, Vy, Vz, ax, ay, az))
#生成观测值
Noise_Process = sigma1 * np.random.randn(IdealState.shape[0], IdealState.shape[1])
Noise_observe = sigma2 * np.random.randn(IdealState.shape[0], IdealState.shape[1])
OberState = IdealState + Noise_observe

#恒定速度，则没有控制输入，则预测方程为 X(k/k-1) = A * X(k-1/k-1)
#定义状态转移矩阵


#观测方程 Z(k) = H*X(k) + V(s),关于位置观测，则观测方程H为：
H = np.eye(IdealState.shape[0])

#生成初始位置
initial_State = OberState[:, 0]

#初始状态协方差
initial_P = np.eye(len(initial_State))

#过程噪声协方差协方差
Qs = np.cov(Noise_Process)

#观测噪声协方差
Rs = np.cov(Noise_observe)

Pk_1 = initial_P

Kaman = IdealState[:, 0].reshape(len(OberState[:, 0]), 1)
Xk = IdealState[:, 0]
for i in range(1, i_iter):
    Xk_1 = Xk.reshape(len(Xk), 1) #前一时刻状态
    Xk_o = OberState[:, i-1].reshape(len(OberState[:, i-1]), 1) #当前时刻的观测值
# (1)预测下一时刻状态
    Xk_p = np.dot(A, Xk_1)
# (2)k时刻的协方差
    Pk_p = np.dot(np.dot(A, Pk_1), A.T) + Qs
# (3)计算k时刻的卡尔曼增益,本例中观测矩阵H为对角阵，即直接观测目标的位置
# kg(k) = P(k/k-1)*H' / (H* P(k/k-1)*H' + R(s) )
    part1 = np.dot(Pk_p, H.T)
    part2 = np.dot(np.dot(H, Pk_p), H.T) + Rs
    kg = np.dot(part1, np.linalg.inv(part2))
# (4)更新状态参数
# X(k/k-1) + kg(k)*(Z(k) - H*X(k/k-1))
    Xk = Xk_p + np.dot(kg, Xk_o - np.dot(H, Xk_p))
# 协方差矩阵更新P(k/k) = P(k/k-1)*(I - kg(k) * H)
    Pk_1 = np.dot(Pk_p, (np.eye(len(initial_State)) - np.dot(kg, H)))
    Kaman = np.hstack((Kaman, Xk))

#理论数据
X_ideal = IdealState[0, :]
Y_ideal = IdealState[1, :]
#观测数据
X_obv = OberState[0, :]
Y_obv = OberState[1, :]
#滤波数据
X_k = Kaman[0, :]
Y_k = Kaman[1, :]

mid1 = (OberState - IdealState) ** 2
mid2 = (Kaman - IdealState) ** 2
deta1 = np.sum(np.sqrt(np.sum(mid1[0:2, :], axis=0)))
deta2 = np.sum(np.sqrt(np.sum(mid2[0:2, :], axis=0)))

plt.plot(X_ideal, Y_ideal,  color='y', ls="-", lw=1)
plt.scatter(X_obv, Y_obv, c='r', marker='o', s=1)
plt.scatter(X_k, Y_k, c='b', marker='o', s=1)
plt.legend()
plt.show()




