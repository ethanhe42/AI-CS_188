# logic_plan_TestClasses.py
# -------------------------
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


# tutorialTestClasses.py
# ----------------------
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


import testClasses

import patrollingGhostAgents
import textDisplay        
import layout
import pacman
import searchAgents 
from search import SearchProblem

# Simple test case which evals an arbitrary piece of python code.
# The test is correct if the output of the code given the student's
# solution matches that of the instructor's.
        

class EvalTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(EvalTest, self).__init__(question, testDict)
        self.preamble = compile(testDict.get('preamble', ""), "%s.preamble" % self.getPath(), 'exec')
        self.test = compile(testDict['test'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        exec self.preamble in bindings
        return str(eval(self.test, bindings))

    def execute(self, grades, moduleDict, solutionDict):
        result = self.evalCode(moduleDict)
        if result == solutionDict['result']:
            grades.addMessage('PASS: %s' % self.path)
            grades.addMessage('\t%s' % self.success)
            return True
        else:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\tstudent result: "%s"' % result)
            grades.addMessage('\tcorrect result: "%s"' % solutionDict['result'])

        return False

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')

        handle.write('result: "%s"\n' % self.evalCode(moduleDict))
        handle.close()
        return True
        
class LogicTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(LogicTest, self).__init__(question, testDict)
        self.preamble = compile(testDict.get('preamble', ""), "%s.preamble" % self.getPath(), 'exec')
        self.test = compile(testDict['test'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        exec self.preamble in bindings
        return eval(self.test, bindings)

    def execute(self, grades, moduleDict, solutionDict):
        result = self.evalCode(moduleDict)
        result = map(lambda x: str(x), result)
        result = ' '.join(result)
        
        if result == solutionDict['result']:
            grades.addMessage('PASS: %s' % self.path)
            grades.addMessage('\t%s' % self.success)
            return True
        else:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\tstudent result: "%s"' % result)
            grades.addMessage('\tcorrect result: "%s"' % solutionDict['result'])

        return False

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')
        solution = self.evalCode(moduleDict)
        solution = map(lambda x: str(x), solution)
        handle.write('result: "%s"\n' % ' '.join(solution))
        handle.close()
        return True        

        
class ExtractActionSequenceTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(ExtractActionSequenceTest, self).__init__(question, testDict)
        self.preamble = compile(testDict.get('preamble', ""), "%s.preamble" % self.getPath(), 'exec')
        self.test = compile(testDict['test'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        exec self.preamble in bindings
        return eval(self.test, bindings)

    def execute(self, grades, moduleDict, solutionDict):
        result = ' '.join(self.evalCode(moduleDict))
        if result == solutionDict['result']:
            grades.addMessage('PASS: %s' % self.path)
            grades.addMessage('\t%s' % self.success)
            return True
        else:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\tstudent result: "%s"' % result)
            grades.addMessage('\tcorrect result: "%s"' % solutionDict['result'])

        return False

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)
        handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')

        handle.write('result: "%s"\n' % ' '.join(self.evalCode(moduleDict)))
        handle.close()
        return True

        
class PositionProblemTest(testClasses.TestCase):
    
    def __init__(self, question, testDict):
        super(PositionProblemTest, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']

    def solution(self, search):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        pac = searchAgents.SearchAgent('plp', 'PositionSearchProblem', search)
        ghosts = []
        disp = textDisplay.NullGraphics()
        games = pacman.runGames(lay, pac, ghosts, disp, 1, False, catchExceptions=True, timeout=1000)
        gameState = games[0].state
        return (gameState.isWin(), gameState.getScore(), pac.actions)

    def execute(self, grades, moduleDict, solutionDict):
        search = moduleDict['search']
        gold_path = solutionDict['solution_path']
        gold_score = int(solutionDict['solution_score'])

        solution = self.solution(search)

        if not solution[0] or solution[1] < gold_score:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\tpacman layout:\t\t%s' % self.layoutName)
            if solution[0]:
                result_str = "wins"
            else:
                result_str = "loses"
            grades.addMessage('\tstudent solution result: Pacman %s' % result_str)
            grades.addMessage('\tstudent solution score: %d' % solution[1])
            grades.addMessage('\tstudent solution path: %s' % ' '.join(solution[2]))
            if solution[1] < gold_score:
                grades.addMessage('Optimal solution not found.')
            grades.addMessage('')
            grades.addMessage('\tcorrect solution score: %d' % gold_score)
            grades.addMessage('\tcorrect solution path: %s' % gold_path)
            return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\tpacman layout:\t\t%s' % self.layoutName)
        grades.addMessage('\tsolution score:\t\t%d' % gold_score)
        grades.addMessage('\tsolution path:\t\t%s' % gold_path)
        return True

    def writeSolution(self, moduleDict, filePath):
        search = moduleDict['search']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        print "Solving problem", self.layoutName
        print self.layoutText

        solution = self.solution(search)

        print "Problem solved"

        handle.write('solution_win: "%s"\n' % str(solution[0]))
        handle.write('solution_score: "%d"\n' % solution[1])
        handle.write('solution_path: "%s"\n' % ' '.join(solution[2]))
        handle.close()
    
        

class FoodProblemTest(testClasses.TestCase):
    
    def __init__(self, question, testDict):
        super(FoodProblemTest, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']

    def solution(self, search):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        pac = searchAgents.SearchAgent('flp', 'FoodSearchProblem', search)
        ghosts = []
        disp = textDisplay.NullGraphics()
        games = pacman.runGames(lay, pac, ghosts, disp, 1, False, catchExceptions=True, timeout=1000)
        gameState = games[0].state
        return (gameState.isWin(), gameState.getScore(), pac.actions)

    def execute(self, grades, moduleDict, solutionDict):
        search = moduleDict['search']
        gold_path = solutionDict['solution_path']
        gold_score = int(solutionDict['solution_score'])

        solution = self.solution(search)

        if not solution[0] or solution[1] < gold_score:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\tpacman layout:\t\t%s' % self.layoutName)
            if solution[0]:
                result_str = "wins"
            else:
                result_str = "loses"
            grades.addMessage('\tstudent solution result: Pacman %s' % result_str)
            grades.addMessage('\tstudent solution score: %d' % solution[1])
            grades.addMessage('\tstudent solution path: %s' % ' '.join(solution[2]))
            if solution[1] < gold_score:
                grades.addMessage('Optimal solution not found.')
            grades.addMessage('')
            grades.addMessage('\tcorrect solution score: %d' % gold_score)
            grades.addMessage('\tcorrect solution path: %s' % gold_path)
            return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\tpacman layout:\t\t%s' % self.layoutName)
        grades.addMessage('\tsolution score:\t\t%d' % gold_score)
        grades.addMessage('\tsolution path:\t\t%s' % gold_path)
        return True

    def writeSolution(self, moduleDict, filePath):
        search = moduleDict['search']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        print "Solving problem", self.layoutName
        print self.layoutText

        solution = self.solution(search)

        print "Problem solved"

        handle.write('solution_win: "%s"\n' % str(solution[0]))
        handle.write('solution_score: "%d"\n' % solution[1])
        handle.write('solution_path: "%s"\n' % ' '.join(solution[2]))
        handle.close()
    
        
class FoodGhostsProblemTest(testClasses.TestCase):
    
    def __init__(self, question, testDict):
        super(FoodGhostsProblemTest, self).__init__(question, testDict)
        self.layoutText = testDict['layout']
        self.layoutName = testDict['layoutName']

    def solution(self, search):
        lay = layout.Layout([l.strip() for l in self.layoutText.split('\n')])
        pac = searchAgents.SearchAgent('fglp', 'FoodGhostsSearchProblem', search)
        ghosts = [patrollingGhostAgents.PatrollingGhost(i) for i in xrange(1,lay.getNumGhosts()+1)]
        disp = textDisplay.NullGraphics()
        games = pacman.runGames(lay, pac, ghosts, disp, 1, False, catchExceptions=True, timeout=1000)
        gameState = games[0].state
        return (gameState.isWin(), gameState.getScore(), pac.actions)

    def execute(self, grades, moduleDict, solutionDict):
        search = moduleDict['search']
        gold_path = solutionDict['solution_path']
        gold_score = int(solutionDict['solution_score'])

        solution = self.solution(search)

        if not solution[0] or solution[1] < gold_score:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\tpacman layout:\t\t%s' % self.layoutName)
            if solution[0]:
                result_str = "wins"
            else:
                result_str = "loses"
            grades.addMessage('\tstudent solution result: Pacman %s' % result_str)
            grades.addMessage('\tstudent solution score: %d' % solution[1])
            grades.addMessage('\tstudent solution path: %s' % ' '.join(solution[2]))
            if solution[1] < gold_score:
                grades.addMessage('Optimal solution not found.')
            grades.addMessage('')
            grades.addMessage('\tcorrect solution score: %d' % gold_score)
            grades.addMessage('\tcorrect solution path: %s' % gold_path)
            return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\tpacman layout:\t\t%s' % self.layoutName)
        grades.addMessage('\tsolution score:\t\t%d' % gold_score)
        grades.addMessage('\tsolution path:\t\t%s' % gold_path)
        return True

    def writeSolution(self, moduleDict, filePath):
        search = moduleDict['search']
        # open file and write comments
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        print "Solving problem", self.layoutName
        print self.layoutText

        solution = self.solution(search)

        print "Problem solved"

        handle.write('solution_win: "%s"\n' % str(solution[0]))
        handle.write('solution_score: "%d"\n' % solution[1])
        handle.write('solution_path: "%s"\n' % ' '.join(solution[2]))
        handle.close()


