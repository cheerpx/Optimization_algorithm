# Import the libs
import numpy as np
import matplotlib.pyplot as plt
import random as rd
# Constant definition
MIN_VAL = [-5.0, -5.0]            # The minmum limit of variable
MAX_VAL = [5.0, 5.0]              # The maxxmum limit of variable


# Class definition
class TS():
    """
        TS class
    """

    def __init__(self, vcount=2, ccount=20, tabuL=25, iters=200, tabu_objs=10):
        """
            Initiate the parameters of TS
            -----------------------------
            Parameter:
                vcount    : The number of variables
                ccount    : The number of candidate solutions
                tabuL     : Tabu length(tenure)
                iters     : Number of iterations
                tabu_objs : Number of tabu objects
        """

        self.vcount = vcount  # The number of variables.
        self.ccount = ccount  # The number of candidate solutions.
        self.iters = iters  # Number of iterations, as the end rule.
        self.tabu_objs = tabu_objs
        self.tabu_list = [None] * self.tabu_objs  # Tabu list, used to store the tabu object.
        self.tabu_len = tabuL  # Tabu length.
        self.tabu_len_list = np.array([0] * self.tabu_objs)  # Tabu length list, Corresponds to the Tabu list.
        self.cur_solu = np.array([0.0] * self.vcount)  # The current solution.
        self.best_solu = np.array([0.0] * self.vcount)  # The best solution.
        self.trace = []  # Record the route of best solution.

    def valuate(self, x):
        """
            valuation function
            ------------------
            The valuation function as the rule of contempt is usually the objective function.
            ------------------
            Parameter:
                x : The solution of the valuation function.
        """

        # Objective function
        value = 5 * np.cos(x[0] * x[1]) + x[0] * x[1] + x[1] ** 3
        # Return value
        return value

    def update_Tabu(self, mode, index=None, solu=None):
        """
            upadte_Tabu function
            --------------------
            This function is used to update the tabu list and the tenure of tabu object.
            --------------------
            Parameter:
                mode  :
                index :
                solu  :
        """

        indices = []  # Store the index the value, which is equal to zero.
        # Update the tenure of tabu object.
        for i in range(len(self.tabu_len_list)):
            if self.tabu_len_list[i] != 0:
                self.tabu_len_list[i] -= 1
                # The ralease mode
        if mode == 'release':
            self.sequence_Tabu(index)
            # The add mode
        elif mode == 'add':
            tabuObj = self.valuate(solu)
            if self.tabu_list[0] == None:
                self.sequence_Tabu(0)
            self.tabu_list[len(self.tabu_list) - 1] = tabuObj
            self.tabu_len_list[len(self.tabu_list) - 1] = self.tabu_len
        # Update the tabu list depending on the content of the tabu_list_list.
        for i in range(len(self.tabu_len_list)):
            if self.tabu_len_list[i] == 0:
                indices.append(i)
        if len(indices) == 1:
            self.sequence_Tabu(indices[0])
        elif len(indices) > 1:
            # Part 1
            maxindex = max(indices)  # Maximum index
            self.sequence_Tabu(maxindex)
            # Part 2
            for i in indices:
                if i != max(indices):
                    self.tabu_list[i] = None  # Set the tabu object as None.
                    self.tabu_len_list[i] = 0  # Set the tenure of tabu object as zero.
            objs = []
            objs1 = []
            for obj in self.tabu_list[:maxindex]:
                if obj != None:
                    objs.append(obj)
            for obj in self.tabu_len_list[:maxindex]:
                if obj != 0:
                    objs1.append(obj)
            if objs != []:
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

    def sequence_Tabu(self, index):
        """
            sequence_Tabu function
            ----------------------
            Parameter:
                index : The index of the tabu object to be deleted.
        """

        if index != len(self.tabu_list) - 1:
            for i in range(len(self.tabu_list) - 1 - index):
                self.tabu_list[index + i] = self.tabu_list[index + i + 1]
                self.tabu_len_list[index + i] = self.tabu_len_list[index + i + 1]
            self.tabu_list[len(self.tabu_list) - 1] = None
            self.tabu_len_list[len(self.tabu_list) - 1] = 0

    def run(self):
        """
            run function
            ------------
            Execute the TS algorithm.
        """

        # Produce the initial solution and set the best soluiton.
        for i in range(self.vcount):
            self.cur_solu[i] = rd.uniform(-2, 2)
            self.best_solu[i] = self.cur_solu[i]
        # Update the tabu list and the tenure of tabu object.
        self.update_Tabu('add', solu=self.cur_solu)
        # Iteration
        counter = 0  # The counter of iteration
        while counter < self.iters:
            counter += 1  # The counter add 1 when finishs a loop.
            candi_solu = np.zeros((self.ccount, self.vcount))  # Store the candidate solutions.
            # Select some candidate solutions from the near area of the current solution.
            for i in range(self.ccount):
                for j in range(self.vcount):
                    candi_solu[i, j] = self.cur_solu[j] + rd.uniform(-1, 1)
            # Identify whether the candidate solutions are kept in the limited area.
            for i in range(self.vcount):
                for j in range(self.ccount):
                    if candi_solu[j, i] > MAX_VAL[i]:
                        candi_solu[j, i] = MAX_VAL[i]
                    elif candi_solu[j, i] < MIN_VAL[i]:
                        candi_solu[j, i] = MIN_VAL[i]
            isAll = False  # A sign of all solutions kept in tabu list.
            isPart = False  # A sign of a part of solutions kept in tabu list.
            count = [0] * self.ccount
            for i in range(self.ccount):
                for k in range(len(self.tabu_list)):
                    if self.valuate(candi_solu[i]) == self.tabu_list[k]:
                        count[i] = 1
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
                #    Part1 : All solutions in Tabu list.   #
                ############################################
                temp_tabu_list = []
                for tabuObj in self.tabu_list:
                    if tabuObj != None:
                        temp_tabu_list.append(tabuObj)
                index = np.argmin(np.array(temp_tabu_list))  # Obtain the index of minimum value from the tabu list
                temp_solu = np.array([0.0] * self.vcount)
                for solu in candi_solu:
                    if self.valuate(solu) == self.tabu_list[index]:
                        temp_solu = solu
                # Update the current solution.
                self.cur_solu = temp_solu
                # Update the best solution according to the valuate function and requirements.
                if self.valuate(self.cur_solu) < self.valuate(self.best_solu):
                    self.best_solu = self.cur_solu
                    # Update the tabu list and the tenure of tabu object.
                self.update_Tabu('release', index=index)

            elif isPart == True:
                ##################################################
                #    Part2 : A part of solutions in Tabu list.   #
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
                    #    Part2.1 : Exist the best solution in  candidate solutions.   #
                    #              Some of these exist in tabu list.                  #
                    ###################################################################
                    isInTabu = False
                    index = 0
                    #
                    if len(temp_bsolu) == 1:
                        bsolu = temp_bsolu[0]
                    elif len(temp_bsolu) != 1 and len(temp_bsolu) != 0:
                        bsolu = temp_bsolu[0]
                        for solu in temp_bsolu[1:]:
                            if self.valuate(solu) < self.valuate(bsolu):
                                bsolu = solu
                    #
                    for i in range(len(self.tabu_list)):
                        if self.valuate(bsolu) == self.tabu_list[i]:
                            isInTabu = True
                            index = i
                    # Update the current solution.
                    self.cur_solu = bsolu
                    # Update the best solution.
                    if self.valuate(bsolu) < self.valuate(self.best_solu):
                        self.best_solu = bsolu
                    #
                    if isInTabu == True:
                        # Update the tabu list and the tenure of tabu object.
                        self.update_Tabu('release', index=index)
                    else:
                        index = len(self.tabu_list) - 1
                        # Update the tabu list and the tenure of tabu object.
                        self.update_Tabu(index, 'add', solu=self.cur_solu)
                else:
                    #################################################################
                    #    Part2.2 : None the best solution in candidate solutions.   #
                    #              None solutions exist in tabu list.               #
                    #################################################################
                    notInTabu = []
                    for solu in candi_solu:
                        count = 0
                        for i in range(len(self.tabu_list)):
                            if self.valuate(solu) != self.tabu_list[i]:
                                count += 1
                        if count == len(self.tabu_list):
                            notInTabu.append(solu)
                    temp_solu = notInTabu[0]
                    if len(notInTabu) != 1:
                        for solu in notInTabu[1:]:
                            if self.valuate(solu) < self.valuate(temp_solu):
                                temp_solu = solu
                    # Update the current solution according to the valuate function and requirements.
                    if self.valuate(temp_solu) < self.valuate(self.cur_solu):
                        self.cur_solu = temp_solu
                        # Update the tabu list and the tenure of tabu object.
                        self.update_Tabu('add', index=len(self.tabu_list) - 1, solu=self.cur_solu)
                        # Update the best solution according to the valuate function and requirements.
                        if self.valuate(self.cur_solu) < self.valuate(self.best_solu):
                            self.best_solu = self.cur_solu

            else:
                #############################################
                #    Part3 : None solutions in tabu list.   #
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


def main():
    """
        main function
    """

    ts = TS(iters=200)
    ts.run()
    print('最优解:', ts.best_solu)
    print('最小值', ts.valuate(ts.best_solu))

    plt.plot(ts.trace, 'r')
    title = 'TS: ' + str(ts.valuate(ts.best_solu))
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    main()