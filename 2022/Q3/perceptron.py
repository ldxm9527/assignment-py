import numpy as np
import matplotlib.pyplot as plt


class Perceptorn(object):

    def __init__(self, l_=0.1) -> None:
        self.w = None
        self.l = l_
        self.n_l = 5e3

    def print_formula(self):
        length = len(self.w)
        formula = [f"{round(self.w[i], 2)} * x{i}" for i in range(1, length)]
        formula = f"{round(self.w[0], 2)} + " + " + ".join(formula) + " = 0"
        length = len(formula)
        print("+" + "-" * length + "+")
        print("|" + formula + "|")
        print("+" + "-" * length + "+")

    def fit1(self, x_, y_):
        # 根据误判点的梯度更新参数,对应a问
        if x_.shape[0] != y_.shape[0]:
            raise ValueError("rows of x and y are not identical, check your input")

        rows = x_.shape[0]
        x_0 = np.ones(rows)
        x_ = np.column_stack((x_0.T, x_))
        self.w = np.zeros(x_.shape[1])

        n = 0  # 迭代次数上限,避免死循环
        while n < self.n_l:
            n += 1
            diff = np.zeros(x_.shape[1])
            for i in range(rows):
                x, y = x_[i], y_[i]
                # 计算是否为误判点,即indicator function里的内容
                if y * (np.dot(self.w, x)) <= 0:
                    diff += np.dot(x, y)
            self.w = self.w + self.l * diff

    def fit2(self, x_, y_):
        # 根据距离最近的点调整超平面,对应b问
        if x_.shape[0] != y_.shape[0]:
            raise ValueError("rows of x and y are not identical, check your input")
        rows = x_.shape[0]
        x_0 = np.ones(rows)
        x_ = np.column_stack((x_0.T, x_))
        self.w = np.zeros(x_.shape[1])

        n = 0  # 迭代次数上限,避免死循环
        while n < self.n_l:
            n += 1
            x_j, y_j = x_[0], y_[0]
            for i in range(1, rows):
                x, y = x_[i], y_[i]
                if y * (np.dot(self.w, x)) < y_j * (np.dot(self.w, x_j)):
                    x_j, y_j = x, y
            self.w = self.w + self.l * np.dot(x_j, y_j)

    def plot(self, x_, y_):
        if x_.shape[1] != 2:
            raise ValueError("dimension doesn't support")
        rows = x_.shape[0]
        x1_pos = [x_[i][0] for i in range(rows) if y_[i] > 0]
        x2_pos = [x_[i][1] for i in range(rows) if y_[i] > 0]
        x1_neg = [x_[i][0] for i in range(rows) if y_[i] < 0]
        x2_neg = [x_[i][1] for i in range(rows) if y_[i] < 0]
        plt.scatter(x1_pos, x2_pos, color = 'hotpink')
        plt.scatter(x1_neg, x2_neg, color = '#88c999')

        xp_1, xp_2 = x_[0][0], x_[0][0]
        for i in range(1, rows):
            if xp_1 > x_[i][0]:
                xp_1 = x_[i][0]
            if xp_2 < x_[i][0]:
                xp_2 = x_[i][0]
        yp_1 = (-self.w[0]-self.w[1] * xp_1)/self.w[2]
        yp_2 = (-self.w[0]-self.w[1] * xp_2)/self.w[2]
        xp = np.array([xp_1, xp_2])
        yp = np.array([yp_1, yp_2])
        plt.plot(xp, yp)
        plt.show()
        return

def gen_point_cloud(need, dim):
    x_pos = np.random.normal(loc=[1, 4], scale=[0.5, 0.75], size=(need//2, dim))
    y_pos = np.ones(need//2)
    x_neg = np.random.normal(loc=[4, 1], scale=[1, 0.75], size=(need-need//2, dim))
    y_neg = (-1) * np.ones(need-need//2)
    x = np.row_stack((x_pos, x_neg))
    y = np.append(y_pos, y_neg)
    return x, y


if __name__ == "__main__":
    x, y = gen_point_cloud(800, 2)
    p = Perceptorn(l_=1)
    p.fit1(x, y)
    p.print_formula()
    p.plot(x, y)
    

    # # 测试,例子来源于《统计学习方法》by李航
    # # 训练集,注意每行x的第一个元素必须是1,以此来代替常数项b
    # x_train = np.array([[1,3,3],[1,4,3],[1,1,1]])
    # y_train = np.array([1,1,-1])

    # p = Perceptorn(l_=1)  # 调整超参数学习率lambda

    # # 调用-fit1
    # p.fit1(x_train, y_train)
    # print("=" * 20 + "[iterate by formula in question A]" + "=" * 20)
    # print(p.w)

    # print()

    # # 调用-fit2
    # p.fit2(x_train, y_train)
    # print("=" * 20 + "[iterate by formula in question B]" + "=" * 20)
    # print(p.w)