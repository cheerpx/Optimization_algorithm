import numpy as np
import matplotlib.pyplot as plt
import random as rd

"""
禁忌搜索算法步骤：
（1）初始化参数：
    产生初始解s，并将禁忌表置为空
（2）根据邻域动作产生邻域解N(s)，根据适应度值选出候选解。
（3）判断候选解是否满足特赦准则，
        （3.1）：如果满足，就将满足特赦准则的解作为当前解，用其对应的对象替换最早进入禁忌表中的对象，更新最优解
        （3.2）：如果不满足特赦准则，就接着判断候选解是否在禁忌表中：
                （3.2.1） 有部分在禁忌表中
                        （3.2.1.1）在候选解中存在比最优方案还好的方案，那就找到候选集中最最优的那个解：
                    # 如果优此时的解于当前最好解，那么就不考虑其是否被禁忌，用这个最好的候选解来更新当前最好
                    # 如果不优于当前最好解，就从所有候选解中选出不在禁忌状态下的最好解作为新的当前解，然后将对应对象加入禁忌表
                        （3.2.1.2）在候选解中不存在比最优方案好的方案，
                        记录不在禁忌表中的候选解，从候选解集合中找到最优的那个解，然后与当前最优解比较，看是否更新当前解与最优解
                （3.2.2） 候选解都不在禁忌表中，那么就将非禁忌的最佳候选解作为当前解，用该解对应的对象替换最早进入禁忌表中的对象
（4）满足终止准则，就输出最优解，如果不满足重复（2）（3）步
"""


# 自变量约束条件
MIN_VAL = [-5.0, -5.0]
MAX_VAL = [5.0, 5.0]


# Class definition
class TS():
    def __init__(self, vcount=2, ccount=20, tabuL=25, iters=200, tabu_objs=10):
        self.vcount = vcount  # 自变量的个数
        self.ccount = ccount  # 候选解的个数
        self.iters = iters  # 最大迭代次数
        self.tabu_objs = tabu_objs  # 禁忌对象
        self.tabu_list = [None] * self.tabu_objs  # 禁忌表，存放禁忌对象,禁忌表长度10
        self.tabu_len = tabuL  # 禁忌长度
        self.tabu_len_list = np.array([0] * self.tabu_objs)  # 禁忌长度列表，对应于禁忌列表。
        self.cur_solu = np.array([0.0] * self.vcount)  # 当前解
        self.best_solu = np.array([0.0] * self.vcount)  # 最优解
        self.trace = []  # 记录最优解的路径

    def valuate(self, x):
        # 目标/评价函数
        value = 5 * np.cos(x[0] * x[1]) + x[0] * x[1] + x[1] ** 3
        return value

    # 更新禁忌表,这一部分是很重要的，对理解代码有着很核心的作用
    # 禁忌表有两种操作：添加禁忌元素和删除禁忌元素
    def update_Tabu(self, mode, index=None, solu=None):
        indices = []  # 存储禁忌对象的禁忌期限为0的对象的索引
        # 更新禁忌对象的禁忌期限
        for i in range(len(self.tabu_len_list)):   # 长度为10，
            if self.tabu_len_list[i] != 0:        # 每进行一次禁忌表更新操作，对应的禁忌对象的使用期限就-1，一直到0
                self.tabu_len_list[i] -= 1
        print("tabu_list", self.tabu_list)
        print("tabu_len_list", self.tabu_len_list)

        # 释放紧急对象
        if mode == 'release':
            self.sequence_Tabu(index)
        # 向禁忌表中添加禁忌对象,添加到禁忌表的
        elif mode == 'add':
            tabuObj = self.valuate(solu)  # 禁忌表对象是解的valuate值,
            if self.tabu_list[0] == None:
                #print("从后往前插入")
                self.sequence_Tabu(0)
            self.tabu_list[len(self.tabu_list) - 1] = tabuObj
            # print("self.tabu_list[9]", self.tabu_list[9])
            self.tabu_len_list[len(self.tabu_list) - 1] = self.tabu_len

        for i in range(len(self.tabu_len_list)):
            if self.tabu_len_list[i] == 0:  # 如果禁忌对象的使用期限为0
                # 就将此禁忌对象的索引加到indices中
                indices.append(i)
        print("禁忌对象的禁忌期限为0的对象的索引:", indices)
        if len(indices) == 1:   # 如果此时indices中只有一个禁忌对象，直接从禁忌表中删除即可
            self.sequence_Tabu(indices[0])
        elif len(indices) > 1:   # 如果此时indices中有多个禁忌对象
            # Part 1
            maxindex = max(indices)     # 找出索引值最大的
            print("索引值最大的", maxindex)
            self.sequence_Tabu(maxindex)  # 然后从禁忌表中删除maxindex
            print("从禁忌表中删除maxindex：", self.tabu_list)
            # Part 2
            for i in indices:          # 遍历indices，里面存放的是禁忌期限为0的索引
                if i != max(indices):     # 如果禁忌对象不等于最大的那个禁忌对象
                    self.tabu_list[i] = None  # 可以直接设置禁忌表中此禁忌对象为None
                    self.tabu_len_list[i] = 0  # 同时将此禁忌对象的禁忌期限设置为0
            objs = []
            objs1 = []
            for obj in self.tabu_list[:maxindex]:
                if obj != None:
                    objs.append(obj)  # objs里存放的是禁忌表中不为None的禁忌对象
            for obj in self.tabu_len_list[:maxindex]:
                if obj != 0:
                    objs1.append(obj)   # objs1里面存放的是禁忌表期限表中不为0的禁忌对象的使用期限
            if objs != []:     #  如果禁忌表中还存在禁忌对象,禁忌对象在禁忌表中往前移动一位
                for i in range(len(objs)):
                    self.tabu_list[maxindex - i - 1] = objs[i]
                    self.tabu_len_list[maxindex - i - 1] = objs1[i]
                for i in range(maxindex - len(objs)):
                    self.tabu_list[i] = None
                    self.tabu_len_list[i] = 0
            else:
                for i in range(maxindex):
                    self.tabu_list[i] = None
                    self.tabu_len_list[i] = 0
            print("此时禁忌表的样子：", self.tabu_list)
            print("此时禁忌表期限表的样子：", self.tabu_len_list)

    def sequence_Tabu(self, index):
        # 从禁忌表中释放禁忌对象
        if index != len(self.tabu_list) - 1:
            for i in range(len(self.tabu_list) - 1 - index):
                self.tabu_list[index + i] = self.tabu_list[index + i + 1]  # 使要删除的元素在禁忌表的最后一位
                self.tabu_len_list[index + i] = self.tabu_len_list[index + i + 1]
            self.tabu_list[len(self.tabu_list) - 1] = None  # 将禁忌表的最后一位置为None，表示删除

            self.tabu_len_list[len(self.tabu_list) - 1] = 0
            #print("后移：", self.tabu_list)
            #print("后移：", self.tabu_len_list)

    def run(self):
        #（1）初始化参数：产生初始解s --> cur_solu ，并将禁忌表置为空
        for i in range(self.vcount):
            # 产生初始解
            self.cur_solu[i] = rd.uniform(-5, 5)
            # 设置最优解
            self.best_solu[i] = self.cur_solu[i]
        print("初始解：", self.cur_solu)   # 初始解： [1.30533424 0.72105434]
        # 更新禁忌列表和禁忌对象的使用期限。
        self.update_Tabu('add', solu=self.cur_solu)
        counter = 0  # 记录迭代次数
        while counter < self.iters:  # 进入迭代循环
            counter += 1  # 当完成一次迭代，就将次数加1

        # （2）根据邻域动作产生邻域解N(s) --> candi_solu
            candi_solu = np.zeros((self.ccount, self.vcount))  # 候选解集合：用来存放候选解
            # 从当前解的邻域内选出一些候选解，加入候选解集合中
            for i in range(self.ccount):
                for j in range(self.vcount):
                    candi_solu[i, j] = self.cur_solu[j] + rd.uniform(-1, 1)   # 邻域动作
            # print("候选解集合：", candi_solu)  # 20行2列
            # 越界保护（几乎所有的算法都要有越界保护，否则有些更新函数会使变量的值跑出约束条件的范围）
            for i in range(self.vcount):
                for j in range(self.ccount):
                    if candi_solu[j, i] > MAX_VAL[i]:
                        candi_solu[j, i] = MAX_VAL[i]
                    elif candi_solu[j, i] < MIN_VAL[i]:
                        candi_solu[j, i] = MIN_VAL[i]

            # A sign of all solutions kept in tabu list.
            # 候选解集合中的候选解是否全部被禁忌，如果为True，表示满足特赦准则
            isAll = False
            # 候选解集合中的候选解是否只有部分在禁忌表中时，如果为True，表示只有部分在禁忌表中
            isPart = False
            count = [0] * self.ccount
            for i in range(self.ccount):  # 遍历候选解集合中的每一个候选解
                for k in range(len(self.tabu_list)):   # 遍历禁忌表
                    # 判断候选解的valuate值是否与禁忌表中某禁忌对象相等
                    if self.valuate(candi_solu[i]) == self.tabu_list[k]: # 如果相等，表示此候选解被禁
                        count[i] = 1  # 将此候选解的标记置为1，表示被禁
            temp = 0
            for i in count:
                if i == 1:
                    temp += 1
            if temp == self.ccount:
                isAll = True
            elif temp < self.ccount and temp > 0:
                isPart = True

            if isAll == True:
                ############################################
                #    Part1 :先来看第一种情况：所有候选解都在禁忌表中，此时满足特赦准则：
                #    将满足特赦准则的解作为当前解，用其对应的对象替换最早进入禁忌表中的对象，更新最优解 #
                ############################################
                temp_tabu_list = []  # 禁忌缓存表
                for tabuObj in self.tabu_list:
                    if tabuObj != None:
                        temp_tabu_list.append(tabuObj)  # 将禁忌表中的元素移到禁忌缓存表中
                # 从禁忌列表中获取最小值的索引
                index = np.argmin(np.array(temp_tabu_list))
                # print("从禁忌列表中获取最小值的索引", index)
                # 临时解
                temp_solu = np.array([0.0] * self.vcount)
                for solu in candi_solu:  # 遍历候选解集合
                    # 如果候选解的评级函数和禁忌表中最小值相等，表示是合中的最佳候选解
                    if self.valuate(solu) == self.tabu_list[index]:
                        temp_solu = solu   # 将此时的候选解赋值给临时解
                # 将该候选解作为当前解
                self.cur_solu = temp_solu
                # Update the best solution according to the valuate function and requirements.
                # 更新最优解：即将当前解的评价函数和最优解的比较，如果当前解更优，就更新最优解
                if self.valuate(self.cur_solu) < self.valuate(self.best_solu):
                    self.best_solu = self.cur_solu
                # Update the tabu list and the tenure of tabu object.
                # 更新禁忌表：从禁忌表中将此解对应的禁忌对象删除。
                self.update_Tabu('release', index=index)

            elif isPart == True:
                ##################################################
                #    Part2 : 候选解有一部分在禁忌表中.   #
                ##################################################
                isExistbest = False
                temp_bsolu = []
                bsolu = np.array([0.0] * self.vcount)
                for solu in candi_solu:
                    if self.valuate(solu) < self.valuate(self.best_solu):
                        isExistbest = True
                        temp_bsolu.append(solu)
                if isExistbest == True:
                    ###################################################################
                    #    Part2.1 : 在候选解决方案中存在最佳解决方案。  #
                    #              有些在禁忌列表中                  #
                    ###################################################################
                    isInTabu = False
                    index = 0
                    # 如果只有一个候选解是优于当前最优解的，那么直接将这个候选解赋值给bsolu
                    if len(temp_bsolu) == 1:
                        bsolu = temp_bsolu[0]
                    # 如果存在多个候选解是优于当前最优解的，那就找出这些候选解中最最优的那一个，赋值给bsolu
                    elif len(temp_bsolu) != 1 and len(temp_bsolu) != 0:
                        bsolu = temp_bsolu[0]
                        for solu in temp_bsolu[1:]:
                            if self.valuate(solu) < self.valuate(bsolu):
                                bsolu = solu
                    # 上面的代码就是 在候选解集合中找到最优的那个解
                    # 已经不满足特赦准则了，那么就要判断候选解是否被禁
                    for i in range(len(self.tabu_list)):  # 遍历禁忌表，看是候选解是否在禁忌表中
                        if self.valuate(bsolu) == self.tabu_list[i]:  # 如果在
                            isInTabu = True
                            index = i  # 记录此解在禁忌表中的位置

                    self.cur_solu = bsolu
                    # 如果优于当前最好解，那么就不考虑其是否被禁忌，用这个最好的候选解来更新当前最好

                    if self.valuate(bsolu) < self.valuate(self.best_solu):
                        self.best_solu = bsolu
                    # 如果不优于当前最好解，就从所有候选解中选出不在禁忌状态下的最好解作为新的当前解，然后将对应对象加入禁忌表
                    if isInTabu == True:
                        # 更新禁忌表和禁忌长度
                        self.update_Tabu('release', index=index)
                    else:
                    # 如果不优于当前最好解，就从所有候选解中选出不在禁忌状态下的最好解作为新的当前解，然后将对应对象加入禁忌表
                        index = len(self.tabu_list) - 1
                        # Update the tabu list and the tenure of tabu object.
                        self.update_Tabu(index, 'add', solu=self.cur_solu)
                else:
                    #################################################################
                    #    Part2.2 : 在候选解决方案中没有一个是最好的解决方案   #
                    #              None solutions exist in tabu list.               #
                    #################################################################
                    notInTabu = []  # 记录不在禁忌表中的候选解
                    for solu in candi_solu:
                        count = 0
                        for i in range(len(self.tabu_list)):
                            if self.valuate(solu) != self.tabu_list[i]:
                                count += 1
                        if count == len(self.tabu_list):
                            notInTabu.append(solu)
                    #
                    temp_solu = notInTabu[0]
                    if len(notInTabu) != 1:  # 有多个候选解，选出最优的那个候选解赋值给temp_solu
                        for solu in notInTabu[1:]:
                            if self.valuate(solu) < self.valuate(temp_solu):
                                temp_solu = solu
                    # 根据适应度值选择是否更新当前解，如果候选解的值比当前解更优就更新当前解
                    if self.valuate(temp_solu) < self.valuate(self.cur_solu):
                        self.cur_solu = temp_solu  # 更新当前解
                        # 更新禁忌表和禁忌长度
                        self.update_Tabu('add', index=len(self.tabu_list) - 1, solu=self.cur_solu)
                        # 根据适应度值更新最优解
                        if self.valuate(self.cur_solu) < self.valuate(self.best_solu):
                            self.best_solu = self.cur_solu

            else:
                #############################################
                #    Part3 : 候选解都不在禁忌表中，即都没有被禁   #
                #############################################
                bcandi_solu = candi_solu[0]
                for solu in candi_solu[1:]:
                    if self.valuate(solu) < self.valuate(bcandi_solu):
                        bcandi_solu = solu
                # Update the current solution according to the valuate function and requirements.
                if self.valuate(bcandi_solu) < self.valuate(self.cur_solu):
                    self.cur_solu = bcandi_solu
                    # Update the tabu list and the tenure of tabu object.
                    self.update_Tabu('add', index=len(self.tabu_list) - 1, solu=self.cur_solu)
                    # Update the best solution according to the valuate function and requirements.
                    if self.valuate(self.cur_solu) < self.valuate(self.best_solu):
                        self.best_solu = self.cur_solu

                        # Add the best solution to the trace list
            self.trace.append(self.valuate(self.best_solu))
            # print(self.trace)


if __name__ == "__main__":
    ts = TS(iters=200)
    ts.run()
    print('最优解:', ts.best_solu)
    print('最小值', ts.valuate(ts.best_solu))
    plt.plot(ts.trace, 'r')
    title = 'TS: ' + str(ts.valuate(ts.best_solu))
    plt.title(title)
    plt.show()
