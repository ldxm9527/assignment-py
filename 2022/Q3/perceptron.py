import numpy as np
import matplotlib.pyplot as plt
import math
import random


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
            raise ValueError(
                "rows of x and y are not identical, check your input")

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
            raise ValueError(
                "rows of x and y are not identical, check your input")
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
        plt.scatter(x1_pos, x2_pos, color='hotpink')
        plt.scatter(x1_neg, x2_neg, color='#88c999')

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


def gen_points(need, center, r, is_pos):
    x = []
    # random angle
    for i in range(need):
        angle = 2 * math.pi * random.random()
        # random radius
        r_ = r * math.sqrt(random.random())
        # calculating coordinates
        x1 = r_ * math.cos(angle) + center[0]
        x2 = r_ * math.sin(angle) + center[1]
        x.append([x1, x2])
    y = np.ones(need) if is_pos else (-1)*np.ones(need)
    return np.array(x), y


def gen_point_cloud(need, center1, r1, center2, r2):
    "只能生成二维的2个点云"
    x_pos, y_pos = gen_points(need//2, center1, r1, True)
    x_neg, y_neg = gen_points(need-need//2, center2, r2, False)
    x = np.row_stack((x_pos, x_neg))
    y = np.append(y_pos, y_neg)
    return x, y


if __name__ == "__main__":
    # 自己输入需要的数据点数量,2个圆心和2个半径
    x, y = gen_point_cloud(800, [-0.75, -0.75], 0.8, [0.75, 0.75], 0.75)
    p = Perceptorn(l_=1)
    # p.fit1(x, y)  # a问的迭代公式
    p.fit2(x, y)  # b问的迭代公式
    p.print_formula()
    p.plot(x, y)
