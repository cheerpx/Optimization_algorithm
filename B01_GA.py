# Step1：按照一定的初始化方法产生初始种群P(t),t=0
# Step2：评价种群P(t)，计算每个个体的适应度值
# Step3：判断是否满足终止条件，如果满足则输出结果；否则转到步骤4
# Step4：按照选择/交叉/变异等遗传算子产生C(t)
# Step5：P(t) = C(t)， 转到步骤2，t=t+1

# 确定问题：求解2*sin(x)+cos(x)的最大值

# Step1:按照一定的初始化方法产生初始种群:其中需要的参数为种群的大小以及 二进制基因型的位数
# 比如x的取值范围时[0,7],那么这个时候x就可以用三位二进制表示
import random
import math
import matplotlib.pyplot as plt


def species_origin(population_size, chromosome_length):
    population = [[]]  # 二维列表，包含染色体和基因
    for i in range(population_size):
        temporary = []  # 染色体暂存器
        for j in range(chromosome_length):
            temporary.append(random.randint(0, 1))  # 随机产生一个染色体，由二进制数组成
        population.append(temporary)  # 将染色体添加到种群中
    return population[1:]  # 将种群返回，种群时一个二维数组，个体和染色体两维


# 编码:将二进制的染色体基因型编码为十进制的表现型
# 所需要的参数：种群以及染色体的长度
def translation(population, chromosome_length):
    temporary = []
    for i in range(len(population)):  # 遍历所有的种群，对于种群中的每一个个体，进行编码
        total = 0
        for j in range(chromosome_length):
            # 编码的方式就是从第一个基因开始，对2求幂再求和，就是二进制转十进制的思想
            total = total + population[i][j] * (math.pow(2, j))
        # 一个染色体编码完成，由一个二进制数编码为了一个十进制数
        temporary.append(total)
    # 返回种群中所有个体编码完成后的十进制数
    return temporary


# Step2：评价种群P(t)，计算每一个个体的适应度值
# 在本例中，函数值总取非负值，以函数最大值为优化目标，所以可以直接将目标函数作为适应度函数。
def function(population, chtomosome_length, max_value):
    temporary = []
    function1 = []
    temporary = translation(population, chtomosome_length)  # 暂存种群中的所有的染色体
    for i in range(len(temporary)):
        # 遍历种群中的每一个个体，temporary[i]表示其中一个个体的十进制数
        # 一个基因代表一个决策变量，其算法是先转化为十进制，然后除以2的基因个数次方-1  （固定公式）
        # 解码
        x = temporary[i] * max_value / (math.pow(2, chtomosome_length) - 1)
        function1.append(2 * math.sin(x) + math.cos(x))
    return function1


# Step3:判断是否满足终止条件,如果不满足进行Step4


# Step4:按照选择/交叉/变异等遗传算子产生子代
# 选择操作
def fitness(function1):
    fitness1 = []
    min_fitness = mf = 0
    for i in range(len(function1)):
        # 在这里就用到了 目标函数值到个体适应度之间的转换关系了，要保证适应度值为非负，这样才能保证被选择的概率非负，
        # 引入mf，是一个相对比较小的数，可以是预先取定的，也可以是进化到当前代为止的最小目标函数值
        if (function1[i] + mf > 0):
            temporary = mf + function1[i]
        else:  # 如果适应度小于0，就定为0
            temporary = 0.0
        fitness1.append(temporary)  # 将适应度添加到列表中
    # 返回种群中每个个体的适应度函数值
    return fitness1


# 计算适应度总和
def sum(fitness1):
    total = 0
    for i in range(len(fitness1)):
        total += fitness1[i]
    return total


# 计算适应度斐波那契列表，这里是为了求出累积的适应度
# 其实这一步就是计算累积概率，为什么呢？上一步计算得到每一个个体被选择的概率时，可以容易看出，总和为1，
# 现在需要将这N个个体按照比例放入0-1范围的线段里，每一个个体应该用多长的线段呢？用线段的长度代表被选择概率
# 因为轮盘赌没有方向，所以再代码中，我们生成一个[0.1]中的随机数，落在0-1的线段上的哪一段位置上的概率就是各个个体被选择的概率。
# https://blog.csdn.net/weixin_39068956/article/details/105121469
def cumsum(fitness1):
    for i in range(len(fitness1) - 2, -1, -1):
        # range(start,stop,[step]) 从len(fitness1)-2  到  0，步长为-1
        # 倒计数
        total = 0
        j = 0
        while (j <= i):
            total += fitness1[j]
            j += 1
        fitness1[i] = total
        fitness1[len(fitness1) - 1] = 1


# 选择种群中个体适应度最大的个体.所需要的参数是种群以及适应度值
def selection(population, fitness1):
    new_fitness = []  # 单个公式暂存器
    # 将所有的适应度求和
    total_fitness = sum(fitness1)
    # 对于每一个个体,将其适应度概率化
    for i in range(len(fitness1)):
        new_fitness.append(fitness1[i] / total_fitness)

    # new_fitness里面现在存放的是累积概率
    cumsum(new_fitness)

    ms = []  # 存活的种群
    # 求出种群的长度
    population_length = pop_len = len(population)
    # 根据随机数确定哪几个能够存活
    for i in range(pop_len):
        # 产生种群个数的随机值，比如种群数为4，那么就产生4个随机数
        ms.append(random.random())  # random() 方法返回随机生成的一个实数，它在[0,1)范围内。
    # 存活的种群排序
    ms.sort()

    fitin = 0  #
    newin = 0  # 用来遍历轮盘赌产生的随机数
    new_population = new_pop = population

    # 轮盘赌方式
    while newin < pop_len:  # 循环次数为种群的大小
        if (ms[newin] < new_fitness[fitin]):
            # 如果产生的随机数小于累积概率，即为被选中的个体，直接将此时的个体赋值给new_pop即可，然后进行下一次轮盘赌选择
            new_pop[newin] = population[fitin]
            newin += 1
        else:  # 如果产生的随机数大于累积概率，就比较下一个累积概率，一直到选出一个个体为止
            fitin += 1
    population = new_pop


# 交叉操作,所需要的参数：种群以及交叉概率
def crossover(population, pcB00):
    pop_len = len(population)

    for i in range(pop_len - 1):
        # 随机生成单点交叉点
        cpoint = random.randint(0, len(population[0]))
        temporary1 = []
        temporary2 = []

        # 将temporary1作为暂存器，暂时存放第i个染色体中的前0到cpoint个基因
        temporary1.extend(population[i][0:cpoint])
        # 然后再把第i+1个染色体中的后cpoint到第i个染色体中的基因个数补充到temporary1后面
        temporary1.extend(population[i + 1][cpoint:len(population[i])])

        # 将temporary2作为暂存器，在那时存放第i+1个染色体中的前0到cpoint个基因
        temporary2.extend(population[i + 1][0:cpoint])
        # 然后再将第i个染色体中的后cpoint到第i个染色体中的基因个数补充到tempporary2后面
        temporary2.extend(population[i][cpoint:len(population[i])])

        # 第i个染色体和第i+1个染色体基因重组。交叉完成
        population[i] = temporary1
        population[i + 1] = temporary2


# 变异算子操作
def mutation(population, pm):
    # 种群中个体的数量
    px = len(population)
    # 个体中基因/染色体的位数
    py = len(population[0])

    for i in range(px):
        if (random.random() < pm):  # 如果小于pm就发生变异
            # 生成0-py-1之间的随机数，就随机产生变异点
            mpoint = random.randint(0, py - 1)
            # 将mpoint个基因进行单点随机变异，变为0或者1
            if (population[i][mpoint] == 1):
                population[i][mpoint] = 0
            else:
                population[i][mpoint] = 1


# 将每一个染色体都转化成十进制 max_value为基因最大值，为了后面画图用
def b2d(b, max_value, chromosome_length):
    total = 0
    for i in range(len(b)):
        total = total + b[i] * math.pow(2, i)
    # 从第一位开始，每一位对2求幂，然后求和，得到十进制数？
    total = total * max_value / (math.pow(2, chromosome_length) - 1)
    return total


# 寻找最好的适应度和个体
def best(population, fitness1):
    px = len(population)
    bestindividual = []
    bestfitness = fitness1[0]

    for i in range(1, px):
        # 循环找出最大的适应度，适应度最大的也就是最好的个体
        if (fitness1[i] > bestfitness):
            bestfitness = fitness1[i]
            bestindividual = population[i]

    return [bestindividual, bestfitness]


# 主程序
population_size = 500
max_value = 10
chromosome_length = 10
pc = 0.6
pm = 0.01
results = [[]]
fitness1 = []
fitmean = []

# 产生一个初始的种群
population = pop = species_origin(population_size, chromosome_length)

# 迭代500次
for i in range(population_size):
    function1 = function(population, chromosome_length, max_value)
    fitness1 = fitness(function1)
    best_individual, best_fitness = best(population, fitness1)
    results.append([best_fitness, b2d(best_individual, max_value, chromosome_length)])

    selection(population, fitness1)
    crossover(population, pc)
    mutation(population, pm)

results = results[1:]
results.sort()
X = []
Y = []
for i in range(500):
    X.append(i)
    Y.append(results[i][0])
print("结束")
plt.plot(X, Y)
plt.show()
