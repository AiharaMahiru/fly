import argparse, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='YouYuan')


class WhaleOptimization():
    """实现鲸鱼优化算法的类
    参考: http://www.alimirjalili.com/WOA.html
    和: https://doi.org/10.1016/j.advengsoft.2016.01.008
    """
    def __init__(self, opt_func, constraints, nsols, b, a, a_step, maximize=False):
        self._opt_func = opt_func  # 目标函数
        self._constraints = constraints  # 约束条件
        self._sols = self._init_solutions(nsols)  # 随机初始化解
        self._b = b  # b 参数
        self._a = a  # a 参数，控制搜索行为
        self._a_step = a_step  # a 参数的减小步长
        self._maximize = maximize  # 是否最大化目标函数
        self._best_solutions = []  # 存储最优解的历史记录
        
    def get_solutions(self):
        """返回解"""
        return self._sols
                                                                  
    def optimize(self):
        """解随机环绕、搜索或攻击"""
        ranked_sol = self._rank_solutions()
        best_sol = ranked_sol[0] 
        # 将最优解包含在下一代解中
        new_sols = [best_sol]
                                                                 
        for s in ranked_sol[1:]:
            if np.random.uniform(0.0, 1.0) > 0.5:                                      
                A = self._compute_A()                                                     
                norm_A = np.linalg.norm(A)                                                
                if norm_A < 1.0:                                                          
                    new_s = self._encircle(s, best_sol, A)                                
                else:                                                                     
                    # 选择随机解
                    random_sol = self._sols[np.random.randint(self._sols.shape[0])]       
                    new_s = self._search(s, random_sol, A)                                
            else:                                                                         
                new_s = self._attack(s, best_sol)                                         
            new_sols.append(self._constrain_solution(new_s))

        self._sols = np.stack(new_sols)
        self._a -= self._a_step

    def _init_solutions(self, nsols):
        """在空间中随机均匀分布初始化解"""
        sols = []
        for c in self._constraints:
            sols.append(np.random.uniform(c[0], c[1], size=nsols))
                                                                            
        sols = np.stack(sols, axis=-1)
        return sols

    def _constrain_solution(self, sol):
        """确保解满足约束条件"""
        constrain_s = []
        for c, s in zip(self._constraints, sol):
            if c[0] > s:
                s = c[0]
            elif c[1] < s:
                s = c[1]
            constrain_s.append(s)
        return constrain_s

    def _rank_solutions(self):
        """找到最优解"""
        fitness = self._opt_func(self._sols[:, 0], self._sols[:, 1])
        sol_fitness = [(f, s) for f, s in zip(fitness, self._sols)]
   
        # 最优解在列表前面
        ranked_sol = list(sorted(sol_fitness, key=lambda x:x[0], reverse=self._maximize))
        self._best_solutions.append(ranked_sol[0])

        return [ s[1] for s in ranked_sol] 

    def print_best_solutions(self):
        print('历代最优解记录')
        print('([适应度], [解])')
        for s in self._best_solutions:
            print(s)
        print('\n')
        print('最优解')
        print('([适应度], [解])')
        print(sorted(self._best_solutions, key=lambda x:x[0], reverse=self._maximize)[0])

    def _compute_A(self):
        '''
        生成一个大小为2的随机数组 r，其值在0.0和1.0之间。
        然后，它计算并返回一个新数组，
        该数组是 _a 数组的每个元素乘以 r 数组的对应元素的2倍，
        减去 _a 数组的对应元素
        '''
        r = np.random.uniform(0.0, 1.0, size=2)
        return (2.0*np.multiply(self._a, r))-self._a

    def _compute_C(self):
        '生成并返回一个大小为2的随机数组，其值在0.0和1.0之间，每个元素乘以2'
        return 2.0*np.random.uniform(0.0, 1.0, size=2)
                                                                 
    def _encircle(self, sol, best_sol, A):
        '''
        接收3个参数：sol，best_sol 和 A。
        它首先调用 _encircle_D 函数计算 D，
        然后返回 best_sol 减去 A 和 D 的按元素乘积
        '''
        D = self._encircle_D(sol, best_sol)
        return best_sol - np.multiply(A, D)
                                                                 
    def _encircle_D(self, sol, best_sol):
        '''
        接收2个参数：sol 和 best_sol。
        它首先调用 _compute_C 函数计算 C，
        然后计算并返回 C 和 best_sol 按元素乘积与 sol 之差的范数
        '''
        C = self._compute_C()
        D = np.linalg.norm(np.multiply(C, best_sol)  - sol)
        return D

    def _search(self, sol, rand_sol, A):
        '''
        接收3个参数：sol，rand_sol 和 A。
        它首先调用 _search_D 函数计算 D，
        然后返回 rand_sol 减去 A 和 D 的按元素乘积
        '''
        D = self._search_D(sol, rand_sol)
        return rand_sol - np.multiply(A, D)

    def _search_D(self, sol, rand_sol):
        '''
        接收2个参数：sol 和 rand_sol。
        它首先调用 _compute_C 函数计算 C，
        然后计算并返回 C 和 rand_sol 按元素乘积与 sol 之差的范数
        '''
        C = self._compute_C()
        return np.linalg.norm(np.multiply(C, rand_sol) - sol)    

    def _attack(self, sol, best_sol):
        '''
        接收2个参数：sol 和 best_sol。
        它首先计算 best_sol 和 sol 之差的范数，并将其赋值给 D。
        然后，它生成一个大小为2的随机数组 L，其值在-1.0和1.0之间。
        最后，它计算并返回一个新数组，该数组是 D 乘以 exp(_b * L)，
        再乘以 cos(2 * pi * L)，最后加上 best_sol。
        '''
        D = np.linalg.norm(best_sol - sol)
        L = np.random.uniform(-1.0, 1.0, size=2)
        return np.multiply(np.multiply(D,np.exp(self._b*L)), np.cos(2.0*np.pi*L))+best_sol

class AnimateScatter():
    """创建一个可更新的动画散点图"""
    '''
    xmin 和 xmax：x 轴的最小值和最大值。
    ymin 和 ymax：y 轴的最小值和最大值。
    pos：散点的初始位置（二维数组）。
    col：散点的颜色。
    func：用于生成背景的函数。
    resolution：生成网格时的间隔。
    t：每次更新散点图之间的时间间隔。
    '''
    def __init__(self, xmin, xmax, ymin, ymax, pos, col, func, resolution, t):
        plt.ion()

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.fig, self.ax = plt.subplots()

        self.c = col  # 散点的颜色
        self.func = func  # 背景函数
        self.t = t  # 更新间隔时间

        # 添加 resolution 以消除空白
        self.x = np.arange(self.xmin, self.xmax + resolution, resolution)
        self.y = np.arange(self.ymin, self.ymax + resolution, resolution)
        xx, yy = np.meshgrid(self.x, self.y, sparse=True)
        self.z = self.func(xx, yy)
        self.update(pos)

    def draw_background(self):
        """绘制填充函数 meshgrid 的轮廓"""
        self.ax.contourf(self.x, self.y, self.z)

    def update_canvas(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, pos):
        self.ax.clear()  # 清空图像
        self.ax.axis([self.xmin, self.xmax, self.ymin, self.ymax])  # 设置坐标轴范围
        self.draw_background()  # 绘制背景
        self.ax.scatter(pos[:, 0], pos[:, 1], s=30, c=self.c)  # 绘制散点
        self.update_canvas()  # 更新画布
        time.sleep(self.t)  # 等待指定时间

def parse_cl_args():
    parser = argparse.ArgumentParser() #创建参数解析器
    parser.add_argument("-nsols", type=int, default=50, dest='nsols', help='每代解决方案数,默认:50')
    parser.add_argument("-ngens", type=int, default=30, dest='ngens', help='迭代代数,默认:20')
    parser.add_argument("-a", type=float, default=2.0, dest='a', help='鲸鱼优化算法特定参数,控制搜索范围,默认:2.0')
    parser.add_argument("-b", type=float, default=0.5, dest='b', help='鲸鱼优化算法特定参数,控制螺旋形,默认:0.5')
    parser.add_argument("-c", type=float, default=None, dest='c', help='绝对解约束值,默认:无,将使用默认约束')
    parser.add_argument("-func", type=str, default='booth', dest='func', help='要优化的函数,默认:booth; 选项:matyas,cross,eggholder,schaffer,booth')
    parser.add_argument("-r", type=float, default=0.25, dest='r', help='函数网格分辨率,默认:0.25')
    parser.add_argument("-t", type=float, default=0.1, dest='t', help='动画睡眠时间,较低的值会增加动画速度,默认:0.1')
    parser.add_argument("-max", default=False, dest='max', action='store_true', help='启用最大化,默认:False(最小化)')


    args = parser.parse_args()   #解析参数
    return args
#从https://en.wikipedia.org/wiki/Test_functions_for_optimization 获取优化函数

def schaffer(X, Y):
    """约束=100,最小值f(0,0)=0"""
    numer = np.square(np.sin(X**2 - Y**2)) - 0.5
    denom = np.square(1.0 + (0.001*(X**2 + Y**2)))

    return 0.5 + (numer*(1.0/denom))

def eggholder(X, Y):
    """约束=512,最小值f(512, 414.2319)=-959.6407"""
    y = Y+47.0
    a = (-1.0)*(y)*np.sin(np.sqrt(np.absolute((X/2.0) + y)))
    b = (-1.0)*X*np.sin(np.sqrt(np.absolute(X-y)))
    return a+b

def booth(X, Y):
    """约束=10,最小值f(1, 3)=0"""
    return ((X)+(2.0*Y)-7.0)**2+((2.0*X)+(Y)-5.0)**2

def matyas(X, Y):
    """约束=10,最小值f(0, 0)=0"""
    return (0.26*(X**2+Y**2))-(0.48*X*Y)

def cross_in_tray(X, Y):
    """约束=10,
    最小值f(1.34941, -1.34941)=-2.06261
    最小值f(1.34941, 1.34941)=-2.06261
    最小值f(-1.34941, 1.34941)=-2.06261
    最小值f(-1.34941, -1.34941)=-2.06261
    """
    B = np.exp(np.absolute(100.0-(np.sqrt(X**2+Y**2)/np.pi)))
    A = np.absolute(np.sin(X)*np.sin(Y)*B)+1
    return -0.0001*(A**0.1)

def levi(X, Y):
    """约束=10,
    最小值f(1,1)=0.0
    """
    A = np.sin(3.0*np.pi*X)**2
    B = ((X-1)**2)*(1+np.sin(3.0*np.pi*Y)**2)
    C = ((Y-1)**2)*(1+np.sin(2.0*np.pi*Y)**2)
    return A + B + C

def main():
    args = parse_cl_args() #解析命令行参数

    nsols = args.nsols  
    ngens = args.ngens  

    funcs = {'schaffer':schaffer, 'eggholder':eggholder, 'booth':booth, 'matyas':matyas, 'cross':cross_in_tray, 'levi':levi}
    func_constraints = {'schaffer':100.0, 'eggholder':512.0, 'booth':10.0, 'matyas':10.0, 'cross':10.0, 'levi':10.0}

    if args.func in funcs:  
        func = funcs[args.func]
    else:
        print('缺失提供的函数'+args.func+'定义。确保函数定义存在或使用命令行选项。')
        return

    if args.c is None:
        if args.func in func_constraints:
            args.c = func_constraints[args.func]
        else:
            print('为提供的函数'+args.func+'缺失约束。在使用前定义约束或通过命令行提供。')
            return

    C = args.c  #约束
    constraints = [[-C, C], [-C, C]]  #约束范围

    opt_func = func   #优化函数

    b = args.b  
    a = args.a  
    a_step = a/ngens  #a值每代衰减量

    maximize = args.max  #启用最大化

    opt_alg = WhaleOptimization(opt_func, constraints, nsols, b, a, a_step, maximize) #初始化鲸鱼算法
    solutions = opt_alg.get_solutions() #获取初始解
    colors = [[1.0, 1.0, 1.0] for _ in range(nsols)]  #初始颜色

    a_scatter = AnimateScatter(constraints[0][0],    #x轴最小值
                            constraints[0][1],    #x轴最大值
                            constraints[1][0],    #y轴最小值
                            constraints[1][1],    #y轴最大值
                            solutions, colors, opt_func, args.r, args.t)  #初始化动画散点图

    best_fitness = []

    for _ in range(ngens):  #迭代ngens代
        opt_alg.optimize()   #执行优化
        solutions = opt_alg.get_solutions()   #获取新解
        best_sol = solutions[0]  #获取最佳解
        best_fitness.append(best_sol[0])  #获取最佳适应度
        a_scatter.update(solutions)  #更新动画散点图

    plt.figure()
    plt.plot(range(ngens), best_fitness)
    plt.xlabel('迭代次数')
    plt.ylabel('最佳适应度')
    plt.title('优化曲线')
    plt.show()

    opt_alg.print_best_solutions() #打印最佳解

if __name__ == '__main__':
    main()
    input('Press ENTER to exit')