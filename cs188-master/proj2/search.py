# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import logic
import game

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostSearchProblem)
        """
        util.raiseNotDefined()

    def terminalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionSearchProblem
        """
        util.raiseNotDefined()

    def result(self, state, action):
        """
        Given a state and an action, returns resulting state and step cost, which is
        the incremental cost of moving to that successor.
        Returns (next_state, cost)
        """
        util.raiseNotDefined()

    def actions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

    def getWidth(self):
        """
        Returns the width of the playable grid (does not include the external wall)
        Possible x positions for agents will be in range [1,width]
        """
        util.raiseNotDefined()

    def getHeight(self):
        """
        Returns the height of the playable grid (does not include the external wall)
        Possible y positions for agents will be in range [1,height]
        """
        util.raiseNotDefined()

    def isWall(self, position):
        """
        Return true if position (x,y) is a wall. Returns false otherwise.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def atLeastOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that at least one of the expressions in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    return logic.Expr("|", *expressions)


def atMostOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that at most one of the expressions in the list is true.
    """
    exact = exactlyOne(expressions)
    none = logic.Expr("&",logic.Expr("~", *expressions))
    return logic.Expr("|", none, exact)


def exactlyOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that exactly one of the expressions in the list is true.
    """
    return logic.Expr("|", *[logic.Expr("&", expr_i, \
        *[logic.Expr("~", expr_j) for expr_j in expressions if expr_i != expr_j]) for expr_i in expressions])


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    plan = []
    i = 0
    while True:
        true_actions = [a for a in actions if logic.PropSymbolExpr(a, i) in model and model[logic.PropSymbolExpr(a, i)]]
        if not true_actions:
            break
        plan.append(true_actions[0])
        i += 1
    return plan

def getPredecessors(problem, extractPos=lambda x: x[0], generateState=lambda x: x):
    """
    Given an instance of a problem - return a dictionary of {location: predecessors} for each
    legal location in the maze
    """
    rows = [[(i+1, j+1) for i in xrange(problem.getWidth()) if not problem.isWall((i+1,j+1))] for j in xrange(problem.getHeight())]
    positions = reduce(lambda x, y: x + y, rows)
    predecessors = {pos:[] for pos in positions}
    for pos in predecessors.keys():
        successors = [(a, problem.result(generateState(pos), a)) for a in problem.actions(generateState(pos))]
        for action, s in successors:
            if extractPos(s) in predecessors:
                predecessors[extractPos(s)].append((action, pos))
    return predecessors

def generateSuccessorState(predecessors={}, time=0):
    actions = [game.Directions.EAST,game.Directions.SOUTH,game.Directions.WEST,game.Directions.NORTH]

    # this is a list of all possible actions, exactlyOne forces us to pick one of them
    t_actions = exactlyOne([logic.PropSymbolExpr(a, time-1) for a in actions])

    if time <= 0:
        return []

    return [exactlyOne([logic.PropSymbolExpr("P",pos[0],pos[1],time) for pos in predecessors.keys()])] +\
           [logic.to_cnf(logic.Expr(">>", logic.PropSymbolExpr("P",pos[0],pos[1],time), \
            exactlyOne([logic.Expr("&", logic.PropSymbolExpr(a, time-1), logic.PropSymbolExpr("P",p[0],p[1],time-1))\
            for (a, p) in preds]))) for (pos, preds) in predecessors.items()] + [logic.to_cnf(t_actions)] +\
           generateSuccessorState(predecessors,time-1)

def positionLogicPlan(problem):
    """
    Given an instance of a PositionSearchProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    actions = [game.Directions.EAST,game.Directions.SOUTH,game.Directions.WEST,game.Directions.NORTH]
    init_t = util.manhattanDistance(problem.getStartState(), problem.getGoalState())
    goal_s = problem.getGoalState()
    preds = getPredecessors(problem)
    start_pos = problem.getStartState()
    init_state = [logic.Expr("&", logic.PropSymbolExpr("P", start_pos[0],start_pos[1],0),\
                *[logic.Expr("~", logic.PropSymbolExpr("P", s[0],s[1],0)) for s in preds.keys() if s != start_pos])]
    for t in xrange(init_t, 51):
        goal = [logic.PropSymbolExpr("P", goal_s[0], goal_s[1]), \
                logic.to_cnf(logic.Expr(">>", logic.PropSymbolExpr("P", goal_s[0], goal_s[1]),\
               logic.Expr("|", *[logic.PropSymbolExpr("P", goal_s[0], goal_s[1], time) for time in xrange(1,t+1)])))]
        successors = generateSuccessorState(preds, t)
        exps = goal + successors + init_state
        model = logic.pycoSAT(exps)
        if model:
            return extractActionSequence(model, actions)
    return []


def foodLogicPlan(problem):
    """
    Given an instance of a FoodSearchProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    # need a list where all the food is
    # go through with the logic phrase, for each food make sure at some time step t I will be there

    actions = [game.Directions.EAST,game.Directions.SOUTH,game.Directions.WEST,game.Directions.NORTH]

    # this is my food grid as a list [(x,y), (x,y) ...]
    food_list = problem.getStartState()[1].asList()

    # this is a list of the distances from my start state to each food on the grid
    manhattan_food_distances = [util.manhattanDistance(problem.getStartState()[0], food) for food in food_list]

    # for the predecessors function
    extractState = lambda x: x[0][0]
    generateState = lambda x: (x, problem.getStartState()[1])

    # return the food that is furthest away
    init_t = max(manhattan_food_distances)

    preds = getPredecessors(problem, extractState, generateState)
    start_pos = problem.getStartState()[0]
    init_state = [logic.Expr("&", logic.PropSymbolExpr("P", start_pos[0],start_pos[1],0),\
                *[logic.Expr("~", logic.PropSymbolExpr("P", s[0],s[1],0)) for s in preds.keys() if s != start_pos])]

    for t in xrange(init_t, 51):
        goal_list = []
        for food in food_list: # food is an (x, y) coordinate
            goal_list.append([logic.PropSymbolExpr("P", food[0], food[1]), \
                logic.to_cnf(logic.Expr(">>", logic.PropSymbolExpr("P", food[0], food[1]),\
               logic.Expr("|", *[logic.PropSymbolExpr("P", food[0], food[1], time) for time in xrange(1,t+1)])))])
        successors = generateSuccessorState(preds, t)

        # makes goal_list a list, previously was a list of lists
        goal_list = reduce(lambda x,y: x+y, goal_list)
        exps = goal_list + successors + init_state
        model = logic.pycoSAT(exps)
        if model:
            return extractActionSequence(model, actions)
    return []

def foodGhostLogicPlan(problem):
    """
    Given an instance of a FoodGhostSearchProblem, return a list of actions that help Pacman
    eat all of the food and avoid patrolling ghosts.
    Ghosts only move east and west. They always start by moving East, unless they start next to
    an eastern wall.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    actions = [game.Directions.EAST,game.Directions.SOUTH,game.Directions.WEST,game.Directions.NORTH]

    # this is my food grid as a list [(x,y), (x,y) ...]
    food_list = problem.getStartState()[1].asList()

    # this is a list of the distances from my start state to each food on the grid
    manhattan_food_distances = [util.manhattanDistance(problem.getStartState()[0], food) for food in food_list]

    # for the predecessors function
    extractState = lambda x: x[0][0]
    generateState = lambda x: (x, problem.getStartState()[1])

    # return the food that is furthest away
    init_t = max(manhattan_food_distances)

    preds = getPredecessors(problem, extractState, generateState)
    start_pos = problem.getStartState()[0]
    init_state = [logic.Expr("&", logic.PropSymbolExpr("P", start_pos[0],start_pos[1],0),\
                *[logic.Expr("~", logic.PropSymbolExpr("P", s[0],s[1],0)) for s in preds.keys() if s != start_pos])]
    ghost_pos_arrays = [getGhostPositionArray(problem, ghost.getPosition()) for ghost in problem.getGhostStartStates()]

    for t in xrange(init_t, 51):
        ghosts = reduce(lambda x,y: x + y, [[[~logic.PropSymbolExpr("P", g[i%len(g)][0],g[i%len(g)][1],i+1),\
                    ~logic.PropSymbolExpr("P", g[i%len(g)][0],g[i%len(g)][1],i)]\
                    for i in xrange(t+1)] for g in ghost_pos_arrays])
        ghosts = reduce(lambda x,y: x + y,ghosts)
        goal_list = []
        for food in food_list: # food is an (x, y) coordinate
            goal_list.append([logic.PropSymbolExpr("P", food[0], food[1]), \
                logic.to_cnf(logic.Expr(">>", logic.PropSymbolExpr("P", food[0], food[1]),\
               logic.Expr("|", *[logic.PropSymbolExpr("P", food[0], food[1], time) for time in xrange(t+1)])))])
        successors = generateSuccessorState(preds, t)

        # makes goal_list a list, previously was a list of lists
        goal_list = reduce(lambda x,y: x+y, goal_list)
        exps = goal_list + successors + init_state + ghosts
        model = logic.pycoSAT(exps)
        if model:
            return extractActionSequence(model, actions)
    return []

def getGhostPositionArray(foodGhostProblem, startPos):
    x,y = startPos
    pos_arr = [(x, y)]
    while not foodGhostProblem.isWall((x+1, y)):
        x += 1
        pos_arr.append((x,y))
    while not foodGhostProblem.isWall((x-1, y)):
        x -= 1
        pos_arr.append((x,y))
    while (x + 1) < startPos[0] and x != startPos[0]:
        x += 1
        pos_arr.append((x,y))
    pos_arr = pos_arr if pos_arr[-1] != startPos else pos_arr[:len(pos_arr)-1]
    return pos_arr


# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan
fglp = foodGhostLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
