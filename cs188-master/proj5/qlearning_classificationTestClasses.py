# qlearning_classificationTestClasses.py
# --------------------------------------
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


# learningTestClasses.py
# ---------------------------
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


import testClasses
import random, math, traceback, sys, os
import layout, textDisplay, pacman, gridworld
import time
from util import Counter, TimeoutFunction, FixedRandom
from collections import defaultdict
from pprint import PrettyPrinter
from hashlib import sha1
import dataClassifier, samples


pp = PrettyPrinter()
VERBOSE = False

import gridworld
LIVINGREWARD = -0.1
NOISE = 0.2

class ApproximateQLearningTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(ApproximateQLearningTest, self).__init__(question, testDict)
        self.discount = float(testDict['discount'])
        self.grid = gridworld.Gridworld(parseGrid(testDict['grid']))
        if 'noise' in testDict: self.grid.setNoise(float(testDict['noise']))
        if 'livingReward' in testDict: self.grid.setLivingReward(float(testDict['livingReward']))
        self.grid = gridworld.Gridworld(parseGrid(testDict['grid']))
        self.env = gridworld.GridworldEnvironment(self.grid)
        self.epsilon = float(testDict['epsilon'])
        self.learningRate = float(testDict['learningRate'])
        self.extractor = 'IdentityExtractor'
        if 'extractor' in testDict:
            self.extractor = testDict['extractor']
        self.opts = {'actionFn': self.env.getPossibleActions, 'epsilon': self.epsilon, 'gamma': self.discount, 'alpha': self.learningRate}
        numExperiences = int(testDict['numExperiences'])
        maxPreExperiences = 10
        self.numsExperiencesForDisplay = range(min(numExperiences, maxPreExperiences))
        self.testOutFile = testDict['test_out_file']
        if maxPreExperiences < numExperiences:
            self.numsExperiencesForDisplay.append(numExperiences)

    def writeFailureFile(self, string):
        with open(self.testOutFile, 'w') as handle:
            handle.write(string)

    def removeFailureFileIfExists(self):
        if os.path.exists(self.testOutFile):
            os.remove(self.testOutFile)

    def execute(self, grades, moduleDict, solutionDict):
        failureOutputFileString = ''
        failureOutputStdString = ''
        for n in self.numsExperiencesForDisplay:
            testPass, stdOutString, fileOutString = self.executeNExperiences(grades, moduleDict, solutionDict, n)
            failureOutputStdString += stdOutString
            failureOutputFileString += fileOutString
            if not testPass:
                self.addMessage(failureOutputStdString)
                self.addMessage('For more details to help you debug, see test output file %s\n\n' % self.testOutFile)
                self.writeFailureFile(failureOutputFileString)
                return self.testFail(grades)
        self.removeFailureFileIfExists()
        return self.testPass(grades)

    def executeNExperiences(self, grades, moduleDict, solutionDict, n):
        testPass = True
        qValuesPretty, weights, actions, lastExperience = self.runAgent(moduleDict, n)
        stdOutString = ''
        fileOutString = "==================== Iteration %d ====================\n" % n
        if lastExperience is not None:
            fileOutString += "Agent observed the transition (startState = %s, action = %s, endState = %s, reward = %f)\n\n" % lastExperience
        weightsKey = 'weights_k_%d' % n
        if weights == eval(solutionDict[weightsKey]):
            fileOutString += "Weights at iteration %d are correct." % n
            fileOutString += "   Student/correct solution:\n\n%s\n\n" % pp.pformat(weights)
        for action in actions:
            qValuesKey = 'q_values_k_%d_action_%s' % (n, action)
            qValues = qValuesPretty[action]
            if self.comparePrettyValues(qValues, solutionDict[qValuesKey]):
                fileOutString += "Q-Values at iteration %d for action '%s' are correct." % (n, action)
                fileOutString += "   Student/correct solution:\n\t%s" % self.prettyValueSolutionString(qValuesKey, qValues)
            else:
                testPass = False
                outString = "Q-Values at iteration %d for action '%s' are NOT correct." % (n, action)
                outString += "   Student solution:\n\t%s" % self.prettyValueSolutionString(qValuesKey, qValues)
                outString += "   Correct solution:\n\t%s" % self.prettyValueSolutionString(qValuesKey, solutionDict[qValuesKey])
                stdOutString += outString
                fileOutString += outString
        return testPass, stdOutString, fileOutString

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            for n in self.numsExperiencesForDisplay:
                qValuesPretty, weights, actions, _ = self.runAgent(moduleDict, n)
                handle.write(self.prettyValueSolutionString('weights_k_%d' % n, pp.pformat(weights)))
                for action in actions:
                    handle.write(self.prettyValueSolutionString('q_values_k_%d_action_%s' % (n, action), qValuesPretty[action]))
        return True

    def runAgent(self, moduleDict, numExperiences):
        agent = moduleDict['qlearningAgents'].ApproximateQAgent(extractor=self.extractor, **self.opts)
        states = filter(lambda state : len(self.grid.getPossibleActions(state)) > 0, self.grid.getStates())
        states.sort()
        randObj = FixedRandom().random
        # choose a random start state and a random possible action from that state
        # get the next state and reward from the transition function
        lastExperience = None
        for i in range(numExperiences):
            startState = randObj.choice(states)
            action = randObj.choice(self.grid.getPossibleActions(startState))
            (endState, reward) = self.env.getRandomNextState(startState, action, randObj=randObj)
            lastExperience = (startState, action, endState, reward)
            agent.update(*lastExperience)
        actions = list(reduce(lambda a, b: set(a).union(b), [self.grid.getPossibleActions(state) for state in states]))
        qValues = {}
        weights = agent.getWeights()
        for state in states:
            possibleActions = self.grid.getPossibleActions(state)
            for action in actions:
                if not qValues.has_key(action):
                    qValues[action] = {}
                if action in possibleActions:
                    qValues[action][state] = agent.getQValue(state, action)
                else:
                    qValues[action][state] = None
        qValuesPretty = {}
        for action in actions:
            qValuesPretty[action] = self.prettyValues(qValues[action])
        return (qValuesPretty, weights, actions, lastExperience)

    def prettyPrint(self, elements, formatString):
        pretty = ''
        states = self.grid.getStates()
        for ybar in range(self.grid.grid.height):
            y = self.grid.grid.height-1-ybar
            row = []
            for x in range(self.grid.grid.width):
                if (x, y) in states:
                    value = elements[(x, y)]
                    if value is None:
                        row.append('   illegal')
                    else:
                        row.append(formatString.format(elements[(x,y)]))
                else:
                    row.append('_' * 10)
            pretty += '        %s\n' % ("   ".join(row), )
        pretty += '\n'
        return pretty

    def prettyValues(self, values):
        return self.prettyPrint(values, '{0:10.4f}')

    def prettyPolicy(self, policy):
        return self.prettyPrint(policy, '{0:10s}')

    def prettyValueSolutionString(self, name, pretty):
        return '%s: """\n%s\n"""\n\n' % (name, pretty.rstrip())

    def comparePrettyValues(self, aPretty, bPretty, tolerance=0.01):
        aList = self.parsePrettyValues(aPretty)
        bList = self.parsePrettyValues(bPretty)
        if len(aList) != len(bList):
            return False
        for a, b in zip(aList, bList):
            try:
                aNum = float(a)
                bNum = float(b)
                # error = abs((aNum - bNum) / ((aNum + bNum) / 2.0))
                error = abs(aNum - bNum)
                if error > tolerance:
                    return False
            except ValueError:
                if a.strip() != b.strip():
                    return False
        return True

    def parsePrettyValues(self, pretty):
        values = pretty.split()
        return values
    

def followPath(policy, start, numSteps=100):
    state = start
    path = []
    for i in range(numSteps):
        if state not in policy:
            break
        action = policy[state]
        path.append("(%s,%s)" % state)
        if action == 'north': nextState = state[0],state[1]+1
        if action == 'south': nextState = state[0],state[1]-1
        if action == 'east': nextState = state[0]+1,state[1]
        if action == 'west': nextState = state[0]-1,state[1]
        if action == 'exit' or action == None:
            path.append('TERMINAL_STATE')
            break
        state = nextState

    return path

def parseGrid(string):
    grid = [[entry.strip() for entry in line.split()] for line in string.split('\n')]
    for row in grid:
        for x, col in enumerate(row):
            try:
                col = int(col)
            except:
                pass
            if col == "_":
                col = ' '
            row[x] = col
    return gridworld.makeGrid(grid)
def computePolicy(moduleDict, grid, discount):
    valueIterator = moduleDict['valueIterationAgents'].ValueIterationAgent(grid, discount=discount)
    policy = {}
    for state in grid.getStates():
        policy[state] = valueIterator.computeActionFromValues(state)
    return policy

EVAL_MULTIPLE_CHOICE=True

numTraining = 100
TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def readDigitData(trainingSize=100, testSize=100):
    rootdata = 'digitdata/'
    # loading digits data
    rawTrainingData = samples.loadDataFile(rootdata + 'trainingimages', trainingSize,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile(rootdata + "traininglabels", trainingSize)
    rawValidationData = samples.loadDataFile(rootdata + "validationimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile(rootdata + "validationlabels", TEST_SET_SIZE)
    rawTestData = samples.loadDataFile("digitdata/testimages", testSize,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", testSize)
    try:
        print "Extracting features..."
        featureFunction = dataClassifier.basicFeatureExtractorDigit
        trainingData = map(featureFunction, rawTrainingData)
        validationData = map(featureFunction, rawValidationData)
        testData = map(featureFunction, rawTestData)
    except:
        display("An exception was raised while extracting basic features: \n %s" % getExceptionTraceBack())
    return (trainingData, trainingLabels, validationData, validationLabels, rawTrainingData, rawValidationData, testData, testLabels, rawTestData)

def readSuicideData(trainingSize=100, testSize=100):
    rootdata = 'pacmandata'
    rawTrainingData, trainingLabels = samples.loadPacmanData(rootdata + '/suicide_training.pkl', trainingSize)
    rawValidationData, validationLabels = samples.loadPacmanData(rootdata + '/suicide_validation.pkl', testSize)
    rawTestData, testLabels = samples.loadPacmanData(rootdata + '/suicide_test.pkl', testSize)
    trainingData = []
    validationData = []
    testData = []
    return (trainingData, trainingLabels, validationData, validationLabels, rawTrainingData, rawValidationData, testData, testLabels, rawTestData)

def readContestData(trainingSize=100, testSize=100):
    rootdata = 'pacmandata'
    rawTrainingData, trainingLabels = samples.loadPacmanData(rootdata + '/contest_training.pkl', trainingSize)
    rawValidationData, validationLabels = samples.loadPacmanData(rootdata + '/contest_validation.pkl', testSize)
    rawTestData, testLabels = samples.loadPacmanData(rootdata + '/contest_test.pkl', testSize)
    trainingData = []
    validationData = []
    testData = []
    return (trainingData, trainingLabels, validationData, validationLabels, rawTrainingData, rawValidationData, testData, testLabels, rawTestData)


smallDigitData = readDigitData(20)
bigDigitData = readDigitData(1000)

suicideData = readSuicideData(1000)
contestData = readContestData(1000)

def tinyDataSet():
    def count(m,b,h):
        c = util.Counter();
        c['m'] = m;
        c['b'] = b;
        c['h'] = h;
        return c;

    training = [count(0,0,0), count(1,0,0), count(1,1,0), count(0,1,1), count(1,0,1), count(1,1,1)]
    trainingLabels = [1,        1,            1           , 1           ,      -1     ,      -1]

    validation = [count(1,0,1)]
    validationLabels = [ 1]

    test = [count(1,0,1)]
    testLabels = [-1]

    return (training,trainingLabels,validation,validationLabels,test,testLabels);


def tinyDataSetPeceptronAndMira():
    def count(m,b,h):
        c = util.Counter();
        c['m'] = m;
        c['b'] = b;
        c['h'] = h;
        return c;

    training = [count(1,0,0), count(1,1,0), count(0,1,1), count(1,0,1), count(1,1,1)]
    trainingLabels = [1,            1,            1,          -1      ,      -1]

    validation = [count(1,0,1)]
    validationLabels = [ 1]

    test = [count(1,0,1)]
    testLabels = [-1]

    return (training,trainingLabels,validation,validationLabels,test,testLabels);


DATASETS = {
    "smallDigitData": lambda: smallDigitData,
    "bigDigitData": lambda: bigDigitData,
    "tinyDataSet": tinyDataSet,
    "tinyDataSetPeceptronAndMira": tinyDataSetPeceptronAndMira,
    "suicideData": lambda: suicideData,
    "contestData": lambda: contestData
}

DATASETS_LEGAL_LABELS = {
    "smallDigitData": range(10),
    "bigDigitData": range(10),
    "tinyDataSet": [-1,1],
    "tinyDataSetPeceptronAndMira": [-1,1],
    "suicideData": ["EAST", 'WEST', 'NORTH', 'SOUTH', 'STOP'],
    "contestData": ["EAST", 'WEST', 'NORTH', 'SOUTH', 'STOP']
}


def getAccuracy(data, classifier, featureFunction=dataClassifier.basicFeatureExtractorDigit):
    trainingData, trainingLabels, validationData, validationLabels, rawTrainingData, rawValidationData, testData, testLabels, rawTestData = data
    if featureFunction != dataClassifier.basicFeatureExtractorDigit:
        trainingData = map(featureFunction, rawTrainingData)
        validationData = map(featureFunction, rawValidationData)
        testData = map(featureFunction, rawTestData)
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    acc = 100.0 * correct / len(testLabels)
    serialized_guesses = ", ".join([str(guesses[i]) for i in range(len(testLabels))])
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (acc)
    return acc, serialized_guesses


class GradeClassifierTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(GradeClassifierTest, self).__init__(question, testDict)

        self.classifierModule = testDict['classifierModule']
        self.classifierClass = testDict['classifierClass']
        self.datasetName = testDict['datasetName']

        self.accuracyScale = int(testDict['accuracyScale'])
        self.accuracyThresholds = [int(s) for s in testDict.get('accuracyThresholds','').split()]
        self.exactOutput = testDict['exactOutput'].lower() == "true"

        self.automaticTuning = testDict['automaticTuning'].lower() == "true" if 'automaticTuning' in testDict else None
        self.max_iterations = int(testDict['max_iterations']) if 'max_iterations' in testDict else None
        self.featureFunction = testDict['featureFunction'] if 'featureFunction' in testDict else 'basicFeatureExtractorDigit'

        self.maxPoints = len(self.accuracyThresholds) * self.accuracyScale


    def grade_classifier(self, moduleDict):
        featureFunction = getattr(dataClassifier, self.featureFunction)
        data = DATASETS[self.datasetName]()
        legalLabels = DATASETS_LEGAL_LABELS[self.datasetName]

        classifierClass = getattr(moduleDict[self.classifierModule], self.classifierClass)

        if self.max_iterations != None:
            classifier = classifierClass(legalLabels, self.max_iterations)
        else:
            classifier = classifierClass(legalLabels)

        if self.automaticTuning != None:
            classifier.automaticTuning = self.automaticTuning

        return getAccuracy(data, classifier, featureFunction=featureFunction)


    def execute(self, grades, moduleDict, solutionDict):
        accuracy, guesses = self.grade_classifier(moduleDict)

        # Either grade them on the accuracy of their classifer,
        # or their exact
        if self.exactOutput:
            gold_guesses = solutionDict['guesses']
            if guesses == gold_guesses:
                totalPoints = self.maxPoints
            else:
                self.addMessage("Incorrect classification after training:")
                self.addMessage("  student classifications: " + guesses)
                self.addMessage("  correct classifications: " + gold_guesses)
                totalPoints = 0
        else:
            # Grade accuracy
            totalPoints = 0
            for threshold in self.accuracyThresholds:
                if accuracy >= threshold:
                    totalPoints += self.accuracyScale

            # Print grading schedule
            self.addMessage("%s correct (%s of %s points)" % (accuracy, totalPoints, self.maxPoints))
            self.addMessage("    Grading scheme:")
            self.addMessage("     < %s:  0 points" % (self.accuracyThresholds[0],))
            for idx, threshold in enumerate(self.accuracyThresholds):
                self.addMessage("    >= %s:  %s points" % (threshold, (idx+1)*self.accuracyScale))

        return self.testPartial(grades, totalPoints, self.maxPoints)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# This is the solution file for %s.\n' % self.path)

        if self.exactOutput:
            _, guesses = self.grade_classifier(moduleDict)
            handle.write('guesses: "%s"' % (guesses,))

        handle.close()
        return True
    
    
