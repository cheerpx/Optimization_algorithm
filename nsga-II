# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm

#Importing required modules
import math
import random
import matplotlib.pyplot as plt

#First function to optimize
def function1(x):
    value = -x**2
    return value

#Second function to optimize
def function2(x):
    value = -(x-2)**2
    return value

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values  给个体排序
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list) != len(list1)):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
# 实现NSGA-II的快速非支配排序，最后返回的是一组非支配解，即Pareto前沿
def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]   # S用来记录种群中被个体p支配的个体的集合，因为有20个个体，所以会有20个[]
    front = [[]]
    n = [0 for i in range(0,len(values1))]      # n用来记录种群中支配个体 p 的个体数
    rank = [0 for i in range(0, len(values1))]
    #对种群中每一个个体，都进行n和s集合的计算
    for p in range(0, len(values1)):
        S[p] = []  #最初的时候，对于每一个个体，S[p]都是空的
        n[p] = 0    #最初的时候，对于每一个个体，n[p]都是0
        for q in range(0, len(values1)):
        # 然后开始对每一个个体p，都遍历种群，找出其支配的个体，存到S中，找到支配p的个体，将其数量存在n中
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                #这个判断关系是说 p支配q，因此可以将q存在S[p]中，前提是满足单一性。
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                #否则的话就是p被支配，n计数+1
                n[p] = n[p] + 1
        if n[p] == 0:  #如果n[p] = 0，说明p是一个非支配解，将它的rank的级别设为最低，这样后面的虚拟适应度值才会最高
            rank[p] = 0
            if p not in front[0]:  #同时将此解加入到Pareto前沿中，此时front[0]中都是最开始产生的非支配解，我们可以将其称之为F1
                front[0].append(p)

    i = 0
    while(front[i] != []): #记住这个循环条件，是用来判断整个种群有咩有被全部分级的。
        Q = []
        for p in front[i]:
            for q in S[p]:
            #对于当前集合F1中的每一个个体p其所支配的个体集合s[p],遍历s[p]中的每一个个体q，执行n[q]-1，如果此时nq=0，就将个体q保存在集合Q中，同时q的等价+1
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        #记F1中得到的个体，即front[0]中的个体为第一个非支配层的个体，并以Q为当前集合，重复上述操作，一直到整个种群被分级。
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
# 拥挤因子
def crowding_distance(values1, values2, front):
    #crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution[i][:])
    #来看看  同属于一个非支配层的两个个体之间的 拥挤度是怎么计算的

    #先取出当前非支配层的所有个体，一共有len(front)
    distance = [0 for i in range(0, len(front))]
    #对于每一个目标函数
    # （1）基于该目标函数对种群进行排序
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    #（2）令边界的两个个体拥挤度位无穷
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    #（3）distance[k]从第二个个体开始，计算distance[k] = distance[k] + [(fm(i+1)-fm(i-1))] / (max - min)
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    #最后返回distance集合，里面存放的是每一个非支配层的每一个个体的拥挤度。
    return distance

#Function to carry out the crossover
# 交叉算子
def crossover(a,b):
    r=random.random()
    if r>0.5:
        return mutation((a+b)/2)
    else:
        return mutation((a-b)/2)

#Function to carry out the mutation operator
# 变异算子
def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob <1:
        solution = min_x+(max_x-min_x)*random.random()
    return solution

#Main program starts here
pop_size = 20
max_gen = 921

#Initialization
min_x = -55
max_x = 55
#产生初始种群，范围在【-55，55】之间，这里只考虑了一个维度，因为只有一个自变量x
solution = [min_x+(max_x-min_x)*random.random() for i in range(0, pop_size)]
print("solutiob:", solution)
gen_no = 0
while (gen_no < max_gen):

    #计算每一迭代过程中的初始种群的适应度值
    function1_values = [function1(solution[i])for i in range(0, pop_size)]
    function2_values = [function2(solution[i])for i in range(0, pop_size)]
    print("function_values_size", len(function1_values))   #20
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
    #返回的是当前代的非支配解的分级表示。
    print("当前非支配解分级的个数：", len(non_dominated_sorted_solution))  #分级的个数是不断在改变的，一直到最后只有一个
    print("The best front for Generation number ", gen_no, " is")
    for valuez in non_dominated_sorted_solution[0]:
        #遍历第一级别的非支配解个体
        print(round(solution[valuez], 3), end=" ")  #round() 方法返回浮点数x的四舍五入值。3是保留3位小数

    #接下来，拥挤度计算与比较
    crowding_distance_values = []
    for i in range(0, len(non_dominated_sorted_solution)):#遍历每一个非支配层
        #non_dominated_sorted_solution[i][:]代表的是第i个非支配层的所有个体
        #首先计算出同属于一个非支配层的个体i 其他个体之间的欧几里得距离，然后将其存在
        crowding_distance_values.append(crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution[i][:]))
    #每一个非支配层都应该有一个拥挤度的
    print("拥挤度的大小", len(crowding_distance_values))
    print("\n")

    solution2 = solution[:]  # 此时当第一次执行时，此时的solution2和初始种群一样
    print("solutiob2:", solution2)

    #Generating offsprings 生成子种群
    while(len(solution2) != 2 * pop_size): #当当前种群的数量不是原本的2倍的时候，表示现在还没有产生子代，那么就产生子代
        a1 = random.randint(0, pop_size-1)
        b1 = random.randint(0, pop_size-1)
        solution2.append(crossover(solution[a1],solution[b1]))
    # print("此时solution2的大小：", len(solution2)) 已经翻倍

    # 此时子父代已经合并了，但是还没有生成新的父种群
    function1_values2 = [function1(solution2[i])for i in range(0, 2*pop_size)]
    function2_values2 = [function2(solution2[i])for i in range(0, 2*pop_size)]
    # 对合并过后的种群进行快速非支配排序以及拥挤度的计算
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
    crowding_distance_values2 = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))
    # 然后选择合适的个体组成新的父种群，每一迭代的过程都是如此
    # 通过快速非支配排序以及拥挤度计算之后，种群中的每一个个体n都得到两个属性，非支配序rank  和 拥挤度 distance
    new_solution = []
    for i in range(0, len(non_dominated_sorted_solution2)):  #遍历每一个非支配层
        #
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in range(0, len(non_dominated_sorted_solution2[i]))]
        # print("得到第", i, "非支配层的解个体的索引", non_dominated_sorted_solution2_1)
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        # print("对其按照拥挤度进行排序：", front22)
        # 遍历当前非支配层中的每一个个体j，
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0, len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = [solution2[i] for i in new_solution]
    gen_no = gen_no + 1

# Lets plot the final front now
# 加负号是将最大值问题转化为最小值问题，即求 x**2  和 （x-2）**2 的最小值
function1 = [i * -1 for i in function1_values]
function2 = [j * -1 for j in function2_values]

plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()
