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
import copy

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

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
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

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this, but you may find it useful for Q5.
    """
    frontier = util.Queue()
    startState = problem.getStartState()

    # Initialize froniter.
    frontier.push((startState,[]))
    expanded = set()
    visited = set()
    # Iterate through the frontier, based on FIFO queue.
    while not frontier.isEmpty():
        node,actionsList = frontier.pop() #(state),[actionsList]
        expanded.add(node)
        if problem.isGoalState(node):
            return actionsList
        else:
            for children in problem.getSuccessors(node):
                if children[0] not in expanded or children[0] not in visited:
                    visited.add(children[0])
                    frontier.push((children[0], actionsList + [children[1]]))
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth.

    Begin with a depth of 1 and increment depth by 1 at every step.
    """
    i = 1
    while True:
        possibleSolution = limitedDepthFirstSearch(problem, set(), set([problem.getStartState()]), problem.getStartState(), i)
        if possibleSolution:
            return possibleSolution
        i += 1

def limitedDepthFirstSearch(problem, expanded, visited, state, depth):
    if depth > 0:
        if problem.isGoalState(state):
            return []
        frontier = util.Stack()
        for s in problem.getSuccessors(state):
            if not s[0] in expanded and s[0] not in visited:
                frontier.push(s)
                visited.add(s[0])
        expanded.add(state)
        while not frontier.isEmpty():
            node = frontier.pop()
            solution = limitedDepthFirstSearch(problem, expanded, visited, node[0], depth - 1)
            if solution != None:
                return [node[1]] + solution
    else :
        return [] if problem.isGoalState(state) else None


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    frontier = util.PriorityQueue()
    frontier.push((problem.getStartState(), [], 0), priority = heuristic(problem.getStartState(), problem = problem))
    expanded = set()
    while not frontier.isEmpty():
        (state, moves, score) = frontier.pop()
        if state in expanded:
            continue
        if problem.isGoalState(state):
            return moves
        expanded.add(state)
        dist = score - heuristic(state, problem = problem)
        successors = [ s for s in problem.getSuccessors(state) if not s[0] in expanded ]
        for successor in successors:
            s_state, s_move, s_cost = successor
            s_h = dist + s_cost + heuristic(s_state, problem = problem)
            frontier.push((s_state, moves + [s_move], s_h), priority = s_h)
    return []

# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
