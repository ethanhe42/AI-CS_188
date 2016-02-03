# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
import time

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state
        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

class Node:
    def __init__(self, state, path, priority):
        self.state = state
        self.path = path
        self.priority = priority

def baseSearch(problem, queue, get_heuristic=None, heuristic=None):
    if isinstance(queue, util.Stack) or isinstance(queue, util.Queue):
        queue.push(Node(problem.getStartState(), [], 0))
    else:
        queue.push(Node(problem.getStartState(), [], 0), 0)
    visited = []

    while not queue.isEmpty():
        # get a new state
        node = queue.pop()

        # process the node if not yet visited
        if node.state in visited:
            continue
        visited.append(node.state)

        if problem.isGoalState(node.state):
            # return path to here
            return node.path
        else:
            # put in the stack the successors if not yet visited
            for s in problem.getSuccessors(node.state):
                if s[0] not in visited:
                    if isinstance(queue, util.Stack) or isinstance(queue, util.Queue):
                        queue.push(Node(s[0], node.path + [s[1]], 0))
                    else:
                        h = get_heuristic(s, node, heuristic, problem)
                        queue.push(Node(s[0], node.path + [s[1]],  node.priority + s[2]), h)
    return None


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    return baseSearch(problem, util.Stack())

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    return baseSearch(problem, util.Queue())

def unif_h(succ, node, heuristic, problem):
    return succ[2] + node.priority

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    return baseSearch(problem, util.PriorityQueue(), unif_h)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def astar_h(succ, node, heuristic, problem):
    return node.priority + succ[2] + heuristic(succ[0], problem)

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    queue = util.PriorityQueue()
    return baseSearch(problem, util.PriorityQueue(), astar_h, heuristic)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch