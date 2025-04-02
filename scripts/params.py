import numpy as np

alpha_0 = 0
a_0 = 0
d_1 = 243.3

alpha_1 = np.pi / 2
a_1 = 0
d_2 = 0

alpha_2 = np.pi
a_2 = 280
d_3 = -10

alpha_3 = np.pi / 2
a_3 = 0
d_4 = 245

alpha_4 = np.pi / 2
a_4 = 0
d_5 = 57

alpha_5 = 0
a_5 = 0
d_6 = 235

DoF = 6

class joint:
    def __init__(self, alpha, a, theta, d):
        self.alpha = alpha
        self.a = a
        self.theta = theta
        self.d = d

    def set_alpha(self, alpha):
        self.alpha = alpha
    
    def set_a(self, a):
        self.a = a

    def set_theta(self, theta):
        self.theta = theta

    def set_d(self, d):
        self.d = d
        

class arm:
    def __init__(self):
        self.joints = []
        for _ in range(DoF):
            self.joints.append(joint(0,0,0,0))

        self.__ALPHA = [alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5]
        self.__A = [a_0, a_1, a_2, a_3, a_4, a_5]
        self.__D = [d_1, d_2, d_3, d_4, d_5, d_6]

        self.__set_other_params(ALPHA=self.__ALPHA, A=self.__A, D=self.__D)


    def set_target_theta(self, THETA, is_Deg=False):
        factor = np.pi / 180 if is_Deg else 1
        for i in range(DoF):
            self.joints[i].set_theta(THETA[i]*factor)

    def __set_other_params(self, ALPHA, A, D):
        for i in range(DoF):
            self.joints[i].set_alpha(ALPHA[i])
            self.joints[i].set_a(A[i])
            self.joints[i].set_d(D[i])

    def print_params(self):
        for i in range(DoF):
            print(self.joints[i].alpha, self.joints[i].a, self.joints[i].theta, self.joints[i].d)

    def __transfer_matrix(self, i):
        '''
        Transfer Matrix
            i 
                T 
            i+1
        '''
        T = np.zeros([4,4])
        print(self.joints[i].theta)
        T[0,0] = np.cos(self.joints[i].theta)
        T[0,1] = -1 * np.sin(self.joints[i].theta)
        T[0,2] = 0
        T[0,3] = self.joints[i].a

        T[1,0] = np.cos(self.joints[i].alpha) * np.sin(self.joints[i].theta)
        T[1,1] = np.cos(self.joints[i].theta) * np.cos(self.joints[i].alpha)
        T[1,2] = -1 * np.sin(self.joints[i].alpha)
        T[1,3] = -1 * np.sin(self.joints[i].alpha) * self.joints[i].d

        T[2,0] = np.sin(self.joints[i].alpha) * np.sin(self.joints[i].theta)
        T[2,1] = np.cos(self.joints[i].theta) * np.sin(self.joints[i].alpha)
        T[2,2] = np.cos(self.joints[i].alpha)
        T[2,3] = np.cos(self.joints[i].alpha) * self.joints[i].d

        T[3,0] = 0
        T[3,1] = 0
        T[3,2] = 0
        T[3,3] = 1

        return T
    
    def T_build(self, is_print = False):
        '''
        The Tool to Base transfer matrix
            Base
                T
            Tool
        '''
        result = np.identity(4)
        for i in range(DoF):
            result = result @ self.__transfer_matrix(i)

        if is_print:
            T = list(result)
            for i in range(4):
                for j in range(4):
                    T[i][j] = round(float(T[i][j]),3)
            for i in range(4):
                print()
                for j in range(4):
                    print(str(T[i][j])+'\t', end='')
        return result


if __name__ == '__main__':
    arm = arm()
    # arm.set_target_theta([0,np.pi/2, np.pi/2, np.pi/2, 0, 0])  # Initial theta array
    arm.set_target_theta([0,0,0,0,0,0])  # ZERO
    # arm.set_target_theta([0,345,75,0,300,0], is_Deg=True)  # HOME
    T = arm.T_build(is_print=True)

