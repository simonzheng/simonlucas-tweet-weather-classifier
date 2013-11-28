import collections, util, copy

############################################################
# Problem 1a

ROW_INDEX = 0
COL_INDEX = 1
def create_nqueens_csp(n = 8):
    """
    Return an N-Queen problem on the board of size |n| * |n|.
    You should call csp.add_variable() and csp.add_binary_potential().

    @param n: number of queens, or the size of one dimension of the board.

    @return csp: A CSP problem with correctly configured potential tables
        such that it can be solved by a weighted CSP solver.
    """
    csp = util.CSP()
    # BEGIN_YOUR_CODE (around 7 lines of code expected)
    queens = []
    for i in range(n):
        queens.append('Q_row%d' %i)
        csp.add_variable('Q_row%d' %i, [(i, col) for col in range(n)])
    for q1 in queens:
        for q2 in queens:
            if q1 != q2:
                csp.add_binary_potential(q1, q2, lambda q1, q2 : q1[ROW_INDEX] != q2[ROW_INDEX] and q1[COL_INDEX] != q2[COL_INDEX] and abs(q1[ROW_INDEX] - q2[ROW_INDEX]) != abs(q1[COL_INDEX] - q2[COL_INDEX]))
                # note: checking row_index condition is not really necessary since we set each of them to have their own values
    # END_YOUR_CODE
    return csp

############################################################
# Problem 1

# A backtracking algorithm that solves weighted CSP.
# Usage:
#   search = BacktrackingSearch()
#   search.solve(csp)
class BacktrackingSearch():

    def reset_results(self):
        """
        This function resets the statistics of the different aspects of the
        CSP sovler. We will be using the values here for grading, so please
        do not make any modification to these variables.
        """
        # Keep track of the best assignment and weight found.
        self.optimalAssignment = {}
        self.optimalWeight = 0

        # Keep track of the number of optimal assignments and assignments. These
        # two values should be identical when the CSP is unweighted or only has binary
        # weights.
        self.numOptimalAssignments = 0
        self.numAssignments = 0

        # Keep track of the number of times backtrack() gets called.
        self.numOperations = 0

        # Keey track of the number of operations to get to the very first successful
        # assignment (doesn't have to be optimal).
        self.firstAssignmentNumOperations = 0

        # List of all solutions found.
        self.allAssignments = []

    def print_stats(self):
        """
        Prints a message summarizing the outcome of the solver.
        """
        if self.optimalAssignment:
            print "Found %d optimal assignments with weight %f in %d operations" % \
                (self.numOptimalAssignments, self.optimalWeight, self.numOperations)
            print "First assignment took %d operations" % self.firstAssignmentNumOperations
        else:
            print "No solution was found."

    def get_delta_weight(self, assignment, var, val):
        """
        Given a CSP, a partial assignment, and a proposed new value for a variable,
        return the change of weights after assigning the variable with the proposed
        value.

        @param assignment: A list of current assignment. len(assignment) should
            equal to self.csp.numVars. Unassigned variables have None values, while an
            assigned variable has the index of the value with respect to its
            domain. e.g. if the domain of the first variable is [5,6], and 6
            was assigned to it, then assignment[0] == 1.
        @param var: Index of an unassigned variable.
        @param val: Index of the proposed value with resepct to |var|'s domain.

        @return w: Change in weights as a result of the proposed assignment. This
            will be used as a multiplier on the current weight.
        """
        assert assignment[var] is None
        w = 1.0
        if self.csp.unaryPotentials[var]:
            w *= self.csp.unaryPotentials[var][val]
            if w == 0: return w
        for var2, potential in self.csp.binaryPotentials[var].iteritems():
            if assignment[var2] == None: continue  # Not assigned yet
            w *= potential[val][assignment[var2]]
            if w == 0: return w
        return w

    def solve(self, csp, mcv = False, lcv = False, mac = False):
        """
        Solves the given weighted CSP using heuristics as specified in the
        parameter. Note that unlike a typical unweighted CSP where the search
        terminates when one solution is found, we want this function to find
        all possible assignments. The results are stored in the variables
        described in reset_result().

        @param csp: A weighted CSP.
        @param mcv: When enabled, Monst Constrained Variable heuristics is used.
        @param lcv: When enabled, Least Constraining Value heuristics is used.
        @param mac: When enabled, AC-3 will be used after each assignment of an
            variable is made.
        """
        # CSP to be solved.
        self.csp = csp

        # Set the search heuristics requested asked.
        self.mcv = mcv
        self.lcv = lcv
        self.mac = mac

        # Reset solutions from previous search.
        self.reset_results()

        # The list of domains of every variable in the CSP. Note that we only
        # use the indeces of the values. That is, if the domain of a variable
        # A is [2,3,5], then here, it will be stored as [0,1,2]. Original domain
        # name/value can be obtained from self.csp.valNames[A]
        self.domains = [list(range(len(domain))) for domain in self.csp.valNames]

        # Perform backtracking search.
        self.backtrack([None] * self.csp.numVars, 0, 1)

        # Print summary of solutions.
        self.print_stats()

    def backtrack(self, assignment, numAssigned, weight):
        """
        Perform the back-tracking algorithms to find all possible solutions to
        the CSP.

        @param assignment: A list of current assignment. len(assignment) should
            equal to self.csp.numVars. Unassigned variables have None values, while an
            assigned variable has the index of the value with respect to its
            domain. e.g. if the domain of the first variable is [5,6], and 6
            was assigned to it, then assignment[0] == 1.
        @param numAssigned: Number of currently assigned variables
        @param weight: The weight of the current partial assignment.
        """

        self.numOperations += 1
        assert weight > 0
        if numAssigned == self.csp.numVars:
            # A satisfiable solution have been found. Update the statistics.
            self.numAssignments += 1
            newAssignment = {}
            for var in range(self.csp.numVars):
                newAssignment[self.csp.varNames[var]] = self.csp.valNames[var][assignment[var]]
            self.allAssignments.append(newAssignment)

            if len(self.optimalAssignment) == 0 or weight >= self.optimalWeight:
                if weight == self.optimalWeight:
                    self.numOptimalAssignments += 1
                else:
                    self.numOptimalAssignments = 1
                self.optimalWeight = weight

                # Map indices to real values for each variable
                self.optimalAssignment = newAssignment
                for var in range(self.csp.numVars):
                    self.optimalAssignment[self.csp.varNames[var]] = \
                        self.csp.valNames[var][assignment[var]]

                if self.firstAssignmentNumOperations == 0:
                    self.firstAssignmentNumOperations = self.numOperations
            return

        # Select the index of the next variable to be assigned.
        var = self.get_unassigned_variable(assignment)

        # Obtain the order of which a variable's values will be tried. Note that
        # this stores the indices of the values with respect to |var|'s domain.
        ordered_values = self.get_ordered_values(assignment, var)

        # Continue the backtracking recursion using |var| and |ordered_values|.
        if not self.mac:
            # When arc consistency check is not enabled.
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    assignment[var] = val
                    # AC-3
                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                    assignment[var] = None
        else:
            # Problem 1e
            # When arc consistency check is enabled.
            # BEGIN_YOUR_CODE (around 10 lines of code expected)
            
            for val in ordered_values:
                # set domain = {xi}
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    # Copy old domains and set new domain
                    oldDomains = copy.deepcopy(self.domains)
                    # print 'domains was ', self.domains
                    # print 'assigning var %d with val %d' %(var, val)
                    assignment[var] = val  # Assign the value to the variable
                    # print 'assignment is ', assignment
                    for assigned_var in range(len(assignment)):
                        assigned_val = assignment[assigned_var]
                        if assigned_val != None:
                            # print 'assigned_var %d has val %d' %(assigned_var, assigned_val)
                            self.domains[assigned_var] = [assigned_val]
                            
                    self.arc_consistency_check(var)


                    # print 'after AC-3 domains is ', self.domains

                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                    assignment[var] = None

                    self.domains = oldDomains



            # remember to set domain = {Xi}
            # look at if not self.mac and copy that for loop and just add literally a few more lines
                # those three lines will stay there, but also copy domains (deep copy) and modify using 
                # self.backtrack and set self.domain back to the original domain
                # check for stuff with floats
            # raise Exception("Not implemented yet")
            # END_YOUR_CODE


    def arc_consistency_check(self, var):
        """
        Perform the AC-3 algorithm. The goal is to reduce the size of the
        domain values for the unassigned variables based on arc consistency.

        @param var: The variable whose value has just been set.

        While not required, you can also choose to add return values in this
        function if there's a need.
        """
        # BEGIN_YOUR_CODE (around 17 lines of code expected)
        
        # Assign value to variable and queue it
        queue = collections.deque()
        queue.appendleft(var)
        # Recursively enforce arc consistency for all items in the queue
        while (len(queue) > 0):
            Xi = queue.pop() # Remove Xi from Queue
            
            # for all neighbors of Xi
                # Enforce arc consistency on Xj with respect to Xi .
                # If Domainj changed, add Xj to queue.
            for Xj, potential in self.csp.binaryPotentials[Xi].iteritems():
                addToQueue = False

                # Check if EVERY value of Xj is inconsistent with Xi
                possibleXjValues = copy.deepcopy(self.domains[Xj])

                for Xj_Val in possibleXjValues:
                    numConsistentXiVals = len(self.domains[Xi])

                    for Xi_Val in self.domains[Xi]:
                        isConsistentWeight = potential[Xi_Val][Xj_Val]
                        if isConsistentWeight == 0:
                            numConsistentXiVals -= 1

                    if numConsistentXiVals == 0:
                        addToQueue = True
                        self.domains[Xj].remove(Xj_Val)

                if addToQueue:
                    queue.appendleft(Xj)

        # raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def get_unassigned_variable(self, assignment):
        """
        Given a partial assignment, return the index of a currently unassigned
        variable.

        @param assignment: A list of current assignment. This is the same as
            what you've seen so far.

        @return var: Index of a currently unassigned variable.
        """

        if not self.mcv:
            # Select a variable without any heuristics.
            for var in xrange(len(assignment)):
                if assignment[var] is None: return var
        else:
            # Problem 1c
            # Heuristic: most constrained variable (MCV)
            # Select a variable with the least number of remaining domain values.
            # BEGIN_YOUR_CODE (around 7 lines of code expected)
            
            # raise Exception("Not implemented yet")
            smallestDomainSize = float('inf')
            mostConstrainedVariable = None
            for var in xrange(len(assignment)):
                if assignment[var] is None: 
                    domainSize = 0
                    for val in self.domains[var]:
                         weight = self.get_delta_weight(assignment, var, val)
                         if weight > 0: 
                            domainSize += 1
                    if domainSize <= smallestDomainSize:
                        mostConstrainedVariable = var
                        smallestDomainSize = domainSize
            # print 'returning mcv as ', mostConstrainedVariable
            return mostConstrainedVariable
            
            # END_YOUR_CODE

    def get_ordered_values(self, assignment, var):
        """
        Given an unassigned variable and a partial assignment, return an ordered
        list of indices of the variable's domain such that the backtracking
        algorithm will try |var|'s values according to this order.

        @param assignment: A list of current assignment. This is the same as
            what you've seen so far.
        @param var: The variable that's going to be assigned next.

        @return ordered_values: A list of indeces of |var|'s domain values.
        """

        if not self.lcv:
            # Return an order of value indices without any heuristics.
            return self.domains[var]
        else:
            # Problem 1d
            # Heuristic: least constraining value (LCV)
            # Return value indices in ascending order of the number of additional
            # constraints imposed on unassigned neighboring variables.
            # BEGIN_YOUR_CODE (around 15 lines of code expected)

            consistentValCount = {}
            for val in self.domains[var]:
                consistentValCount[val] = 0
                weight = self.get_delta_weight(assignment, var, val) # Check if the var is consistent with the rest
                if weight == 0:
                    continue
                # Iterate through all the binary potentials 
                for var2, potential in self.csp.binaryPotentials[var].iteritems(): # for all neighbors with a binary potential
                    if assignment[var2] == None: 
                        for var2Val in self.domains[var2]:
                            isConsistentWeight = potential[val][var2Val]
                            if isConsistentWeight > 0 and self.get_delta_weight(assignment, var2, var2Val) > 0:
                                consistentValCount[val] += 1
                    else:
                        consistentValCount[val] += 1
            
            orderedValues = sorted(consistentValCount, key=consistentValCount.get, reverse=True)
            # print 'consistentValCount is ', consistentValCount
            # print 'orderedValues = ', orderedValues
            return orderedValues # add all possible values to count

            # Given the next variable to be assigned Xj, sort its domain values a in
            # descending order of the number of values b of an unassigned variable Xk
            # that are consistent with Xj=a (consistent means the binary potential on
            # Xj=a and Xk=b is non-zero). Note that you should count only values of b
            # which are already consistent with the existing partial assignment. 
            # Implement this heuristic in get_ordered_values() under the condition self.lcv = True.
            # Note that for this function, you will need to use binaryPotentials in CSP.

            # END_YOUR_CODE


############################################################
# Problem 2

def get_or_variable(csp, name, variables, value):
    """
    Create a new variable with domain [True, False] that can only be assigned to
    True iff at least one of the |variables| is assigned to |value|. You should
    add any necessary intermediate variables, unary potentials, and binary
    potentials to achieve this. Then, return the name of this variable.

    @param name: Prefix of all the variables that are going to be added.
        Can be any hashable objects. For every variable |var| added in this
        function, it's recommended to use a naming strategy such as
        ('or', |name|, |var|) to avoid conflicts with other variable names.
    @param variables: A list of variables in the CSP that are participating
        in this OR function. Note that if this list is empty, then the new
        variable created should never be assigned to True.
    @param value: For the main OR variable being created to be assigned to
        True, at least one of these variables must have this value.

    @return result: The OR variable's name. This variable should have domain
        [True, False] and constraints s.t. it's assigned to True iff at least
        one of the |variables| is assigned to |value|.
    """

    # BEGIN_YOUR_CODE (around 20 lines of code expected)

    prevAuxVarName = None
    resultVarName = None

    # Note that if len(variables) == 0: newVar = False
    if len(variables) == 0:
        resultVarName = (name, 'result')
        csp.add_variable(resultVarName, [True, False])
        csp.add_unary_potential(resultVarName, lambda x: x == False)

    for var_index in range(len(variables)):
        # Add the new variable
        var = variables[var_index]
        auxVarName = ('or', name, var)
        auxVarVals = [(x, y) for x in [True, False] for y in [True, False]]
        csp.add_variable(auxVarName, auxVarVals)

        # Add a binary potential between Xi and Ai
        csp.add_binary_potential(var, auxVarName, lambda var, auxVar : auxVar[1] == (auxVar[0] or (var == value)))

        if prevAuxVarName == None:
            # Special case for first variable
            csp.add_unary_potential(auxVarName, lambda auxVar : auxVar[0] == False)
        else:
            # All other variables should be related to previous one
            csp.add_binary_potential(prevAuxVarName, auxVarName, lambda prevAuxVar, auxVar : auxVar[0] == prevAuxVar[1])

        # Last Variable
        if var_index == len(variables) - 1:
            resultVarName = ('or', name)
            resultVarVals = [True, False]
            csp.add_variable(resultVarName, resultVarVals)
            csp.add_binary_potential(resultVarName, auxVarName, lambda resultVar, auxVar : resultVar == auxVar[1])
        prevAuxVarName = auxVarName

    return resultVarName
    # END_YOUR_CODE

def get_sum_variable(csp, name, variables, maxSum):
    """
    Given a list of |variables| each with non-negative integer domains,
    returns the name of a new variable with domain [0, maxSum], such that
    it's consistent with the value |n| iff there exists an assignment for
    |variables| that sums to |n|.

    @param name: Prefix of all the variables that are going to be added.
        Can be any hashable objects. For every variable |var| added in this
        function, it's recommended to use a naming strategy such as
        ('sum', |name|, |var|) to avoid conflicts with other variable names.
    @param variables: A list of variables that are already in the CSP that
        have non-negative integer values as its domain.
    @param maxSum: An integer indicating the maximum sum value allowed.

    @return result: The name of a newly created variable with domain
        [0, maxSum] such that it's consistent with an assignment of |n|
        iff the there exists an assignment of |variables| that sums to |n|.
    """
    prevAuxVarName = None
    resultVarName = None

    # # Note that if len(variables) == 0: newVar = False
    if len(variables) == 0:
        resultVarName = (name, 'result')
        csp.add_variable(resultVarName, range(0, maxSum+1))
        csp.add_unary_potential(resultVarName, lambda x: x == 0)

    for var_index in range(len(variables)):
        # Add the new variable
        var = variables[var_index]
        auxVarName = ('sum', name, var)
        auxVarVals = [(x,y) for x in range(0, maxSum+1) for y in range(0, maxSum+1)]
        csp.add_variable(auxVarName, auxVarVals)

        # Add a binary potential between Xi and Ai
        csp.add_binary_potential(var, auxVarName, lambda var, auxVar : auxVar[1] == (auxVar[0] + var))

        if prevAuxVarName == None:
            # Special case for first variable
            csp.add_unary_potential(auxVarName, lambda auxVar : auxVar[0] == 0)
        else:
            # All other variables should be related to previous one
            csp.add_binary_potential(prevAuxVarName, auxVarName, lambda prevAuxVar, auxVar : auxVar[0] == prevAuxVar[1])

        # Last Variable
        if var_index == len(variables) - 1:
            resultVarName = ('sum', name)
            resultVarVals = range(0, maxSum+1)
            csp.add_variable(resultVarName, resultVarVals)
            csp.add_binary_potential(resultVarName, auxVarName, lambda resultVar, auxVar : resultVar == auxVar[1])
        
        prevAuxVarName = auxVarName
    return resultVarName
    # END_YOUR_CODE


############################################################
# Problem 3

# A class providing methods to generate CSP that can solve the course scheduling
# problem.
class SchedulingCSPConstructor():

    def __init__(self, bulletin, profile):
        """
        Saves the necessary data.

        @param bulletin: Stanford Bulletin that provides a list of courses
        @param profile: A student's profile and requests
        """
        self.bulletin = bulletin
        self.profile = profile

    def add_variables(self, csp):
        """
        Adding the variables into the CSP. Each variable, (req, quarter),
        can take on the value of one of the courses requested in req or None.
        For instance, for quarter='Aut2013', and a request object, req, generated
        from 'CS221 or CS246', then (req, quarter) should have the domain values
        ['CS221', 'CS246', None]. Conceptually, if var is assigned 'CS221'
        then it means we are taking 'CS221' in 'Aut2013'. If it's None, then
        we not taking either of them in 'Aut2013'.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for req in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_variable((req, quarter), req.cids + [None])

    def add_bulletin_constraints(self, csp):
        """
        Add the constraints that a course can only be taken if it's offered in
        that quarter.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for req in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_unary_potential((req, quarter), \
                    lambda cid: cid is None or \
                        self.bulletin.courses[cid].is_offered_in(quarter))

    def add_norepeating_constraints(self, csp):
        """
        No course can be repeated. Coupling with our problem's constraint that
        only one of a group of requested course can be taken, this implies that
        every request can only be satisfied in at most one quarter.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for req in self.profile.requests:
            for quarter1 in self.profile.quarters:
                for quarter2 in self.profile.quarters:
                    if quarter1 == quarter2: continue
                    csp.add_binary_potential((req, quarter1), (req, quarter2), \
                        lambda cid1, cid2: cid1 is None or cid2 is None)

    def get_basic_csp(self):
        """
        Return a CSP that only enforces the basic constraints that a course can
        only be taken when it's offered and that a request can only be satisfied
        in at most one quarter.

        @return csp: A CSP where basic variables and constraints are added.
        """
        csp = util.CSP()
        self.add_variables(csp)
        self.add_bulletin_constraints(csp)
        self.add_norepeating_constraints(csp)
        return csp

    def add_quarter_constraints(self, csp):
        """
        If the profile explicitly wants a request to be satisfied in some given
        quarters, e.g. Aut2013, then add constraints to not allow that request to
        be satisified in any other quarter.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # BEGIN_YOUR_CODE (around 5 lines of code expected)

        for req in self.profile.requests:
            if req.quarters != None and len(req.quarters) != 0:
                undesired_quarters = copy.deepcopy(self.profile.quarters)
                # print 'undesired_quarters is', undesired_quarters
                # print 'desired_quarters is ', req.quarters
                for desired_quarter in req.quarters:
                    undesired_quarters.remove(desired_quarter)
                for undesired_quarter in undesired_quarters:    
                    csp.add_unary_potential((req, undesired_quarter), \
                        lambda cid: cid is None)
        # END_YOUR_CODE

    def add_request_weights(self, csp):
        """
        Incorporate weights into the CSP. By default, a request has a weight
        value of 1 (already configured in Request). You should only use the
        weight when one of the requested course is in the solution. A
        unsatisfied request should also have a weight value of 1.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        for req in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_unary_potential((req, quarter), \
                    lambda cid: req.weight if cid != None else 1.0)
        # END_YOUR_CODE


    def add_prereq_constraints(self, csp):
        """
        Adding constraints to enforce prerequisite. A course can have multiple
        prerequisites. You can assume that all courses in req.prereqs are
        being requested. Note that if our parser inferred that one of your
        requested course has additional prerequisites that are also being
        requested, these courses will be added to req.prereqs. You will be notified
        with a message when this happens. Also note that req.prereqs apply to every
        single course in req.cids. If a course C has prerequisite A that is requested
        together with another course B (i.e. a request of 'A or B'), then taking B does
        not count as satisfying the prerequisite of C. You cannot take a course
        in a quarter unless all of its prerequisites have been taken before that
        quarter. You should take advantage of get_or_variable().

        @param csp: The CSP where the additional constraints will be added to.
        """
        for req in self.profile.requests:
            for reqQuarter in self.profile.quarters:
                for prereq in req.prereqs:
                    # declare new set of variables
                    prereqTakenVariables = [] # used as a parameter for 'or' variable
                    for prereqQuarter in self.profile.quarters:
                        # add the fact that req cannot be taken before prereq
                        # print 'req is ', req, ' with reqQuarter ', reqQuarter, ' and prereq is ', prereq, 'and prereqQuarter ', prereqQuarter
                        for req2 in self.profile.requests:
                            if (self.profile.quarters.index(prereqQuarter) < self.profile.quarters.index(reqQuarter)):
                                prereqTakenVariables.append((req2, prereqQuarter))
                            # make sure that this is before whatever quarter you're trying to take it in
                    # add the or variable
                    or_variable_name = (req, reqQuarter, prereq)
                    targetCID = prereq
                    or_var = get_or_variable(csp, or_variable_name, prereqTakenVariables, targetCID)
                    
                    # add the binary potential
                    csp.add_binary_potential(or_var, (req, reqQuarter), lambda x,y : x == True if y != None else 1.0)

                    # eventually want the or variable to be true, with the binary potential to the requested course
        
        # # Nonrepeatable 
        # for req in self.profile.requests:
        #     for quarter1 in self.profile.quarters:
        #         for quarter2 in self.profile.quarters:
        #             if quarter1 == quarter2: continue
        #             csp.add_binary_potential((req, quarter1), (req, quarter2), \
        #                 lambda cid1, cid2: cid1 is None or cid2 is None)

        # END_YOUR_CODE

    def add_unit_constraints(self, csp):
        """
        Add constraint to the CSP to ensure that the total number of units are
        within profile.minUnits/maxmaxUnits, inclusively. The allowed range for
        each course can be obtained from bulletin.courses[id].minUnits/maxUnits.
        For a request 'A or B', if you choose to take A, then you must use a unit
        number that's within the range of A. You should introduce any additional
        variables that you are needed. In order for our solution extractor to
        obtain the number of units, for every course that you plan to take in
        the solution, you must have a variable named (courseId, quarter) (e.g.
        ('CS221', 'Aut2013') and it's assigned value is the number of units.
        You should take advantage of get_sum_variable().

        @param csp: The CSP where the additional constraints will be added to.
        """
        for quarter in self.profile.quarters:
            quarterUnitsVariables = []
            quarterMinUnits = self.profile.minUnits
            quarterMaxUnits = self.profile.maxUnits
            for req in self.profile.requests:
                for course_id in req.cids:
                    courseMinUnits = self.bulletin.courses[course_id].minUnits
                    courseMaxUnits = self.bulletin.courses[course_id].maxUnits
                    coursePossibleUnits = range(courseMinUnits, courseMaxUnits + 1) + [0]
                    quarterUnitsVariables.append((course_id, quarter))
                    csp.add_variable((course_id, quarter), coursePossibleUnits)
                    csp.add_binary_potential((req,quarter), (course_id,quarter), lambda x,y: y!=0 if x==course_id else y==0 )
            
            sum_var_name = quarter
            sum_var = get_sum_variable(csp, sum_var_name, quarterUnitsVariables, quarterMaxUnits)
            csp.add_unary_potential(sum_var, lambda sum : sum <= quarterMaxUnits and sum >= quarterMinUnits)

    def add_all_additional_constraints(self, csp):
        """
        Add all additional constraints to the CSP.

        @param csp: The CSP where the additional constraints will be added to.
        """
        self.add_quarter_constraints(csp)
        self.add_request_weights(csp)
        self.add_prereq_constraints(csp)
        self.add_unit_constraints(csp)
