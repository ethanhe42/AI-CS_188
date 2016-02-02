# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        farest_food = 0
        nearest_food = 99999
        if newFood.count() > 0:
            for food in newFood.asList():
                farest_food = max(farest_food, manhattanDistance(newPos, food))
                nearest_food = min(nearest_food, manhattanDistance(newPos, food))
        else:
            nearest_food = 0

        c = 0
        for capsule in successorGameState.getCapsules():
            c = max(c, manhattanDistance(newPos, capsule))

        ng = 99999
        fg = 0
        for ghost in newGhostStates:
            ng = min(ng, manhattanDistance(newPos, ghost.getPosition()))
            fg = max(fg, manhattanDistance(newPos, ghost.getPosition()))

        evaluation = successorGameState.getScore() - newFood.count(False) - nearest_food + ng
        return evaluation#successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction()

        pacman_actions = gameState.getLegalActions(0)
        self.agents = gameState.getNumAgents()
        scores = []
        states = []
        for action in pacman_actions:
            self.cur_depth = 1
            self.cur_agent = 0
            state = gameState.generateSuccessor(0, action)
            states.append(state)
            scores.append(self.get_value(state))
        best_score = max(scores)
        best_indices = [i for i in range(len(scores)) if scores[i] == best_score] # find best actions
        return pacman_actions[random.choice(best_indices)]               # choose one of the best choices avoiding
        # something terrible

    def get_value(self, state):
        if self.cur_depth > self.depth * self.agents - 1\
                or state.isWin() or state.isLose():            # if win or lose return back
            v = self.evaluationFunction(state)
            return v

        self.cur_depth += 1

        self.cur_agent += 1
        if self.cur_agent >= self.agents:
            self.cur_agent = 0

        successors = [state.generateSuccessor(self.cur_agent, action)
                      for action in state.getLegalActions(self.cur_agent)]

        if self.cur_agent == 0:
            v = self.get_max_value(successors)
            self.cur_depth -= 1
            self.cur_agent = self.agents - 1
            return v
        else:
            v = self.get_min_value(successors)
            self.cur_depth -= 1
            self.cur_agent -= 1
            return v

    def get_max_value(self, successors):
        v = -99999
        for successor in successors:
            v = max(v, self.get_value(successor))
        return v

    def get_min_value(self, successors):
        v = 99999
        for successor in successors:
            v = min(v, self.get_value(successor))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction()

        pacman_actions = gameState.getLegalActions(0)
        self.agents = gameState.getNumAgents()
        v = -99999
        alpha = -99999
        beta = 99999
        next_action = Directions.STOP
        for action in pacman_actions:
            self.cur_depth = 1
            self.cur_agent = 0
            state = gameState.generateSuccessor(0, action)
            score = self.get_value(state, alpha, beta)

           # if score > beta:
           #     return action

            if score > v:
                v = score
                next_action = action

            alpha = max(v, alpha)

        return next_action

    def get_value(self, state, alpha, beta):
        if self.cur_depth > self.depth * self.agents - 1\
                or state.isWin() or state.isLose():
            v = self.evaluationFunction(state)
            return v

        self.cur_depth += 1

        self.cur_agent += 1
        if self.cur_agent >= self.agents:
            self.cur_agent = 0

        if self.cur_agent == 0:
            v = self.get_max_value(state, alpha, beta)
            self.cur_depth -= 1
            self.cur_agent = self.agents - 1
            return v
        else:
            v = self.get_min_value(state, alpha, beta)
            self.cur_depth -= 1
            self.cur_agent -= 1
            return v

    def get_max_value(self, state, alpha, beta):
        v = -99999
        for action in state.getLegalActions(self.cur_agent):
            successor = state.generateSuccessor(self.cur_agent, action)
            v = max(v, self.get_value(successor, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def get_min_value(self, state, alpha, beta):
        v = 99999
        for action in state.getLegalActions(self.cur_agent):
            successor = state.generateSuccessor(self.cur_agent, action)
            v = min(v, self.get_value(successor, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """


    def getAction(self, gameState):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction()

        pacman_actions = gameState.getLegalActions(0)
        self.agents = gameState.getNumAgents()
        v = -99999
        next_action = Directions.STOP
        for action in pacman_actions:
            self.cur_depth = 1
            self.cur_agent = 0
            state = gameState.generateSuccessor(0, action)
            score = self.get_value(state)

            if score > v:
                v = score
                next_action = action

        return next_action

    def get_value(self, state):
        if self.cur_depth > self.depth * self.agents - 1\
                or state.isWin() or state.isLose():
            v = self.evaluationFunction(state)
            return v

        self.cur_depth += 1

        self.cur_agent += 1
        if self.cur_agent >= self.agents:
            self.cur_agent = 0

        if self.cur_agent == 0:
            v = self.get_max_value(state)
            self.cur_depth -= 1
            self.cur_agent = self.agents - 1
            return v
        else:
            v = self.get_min_value(state)
            self.cur_depth -= 1
            self.cur_agent -= 1
            return v

    def get_max_value(self, state):
        v = -99999
        for action in state.getLegalActions(self.cur_agent):
            successor = state.generateSuccessor(self.cur_agent, action)
            v = max(v, self.get_value(successor))

        return v

    def get_min_value(self, state):
        v = 0
        actions=state.getLegalActions(self.cur_agent)
        num_actions=float(len(actions))
        for action in actions:
            successor = state.generateSuccessor(self.cur_agent, action)
            v +=self.get_value(successor)

        return v/num_actions

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    farest_food = 0
    nearest_food = 99999
    if newFood.count() > 0:
        for food in newFood.asList():
            farest_food = max(farest_food, manhattanDistance(newPos, food))
            nearest_food = min(nearest_food, manhattanDistance(newPos, food))
    else:
        nearest_food = 0

    c = 0
    for capsule in currentGameState.getCapsules():
        c = max(c, manhattanDistance(newPos, capsule))

    ng = 99999
    fg = 0
    for ghost in newGhostStates:
        ng = min(ng, manhattanDistance(newPos, ghost.getPosition()))
        fg = max(fg, manhattanDistance(newPos, ghost.getPosition()))

    evaluation = currentGameState.getScore() - newFood.count(False) - nearest_food + ng
    return evaluation

# Abbreviation
better = betterEvaluationFunction

