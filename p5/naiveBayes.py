# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.
    
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        
    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """    
            
        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
        
        if (self.automaticTuning):
                kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
                kgrid = [self.k]
                
        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
            
    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter 
        that gives the best accuracy on the held-out validationData.
        
        trainingData and validationData are lists of feature Counters.    The corresponding
        label lists contain the correct label for each datum.
        
        To get the list of all possible features or labels, use self.features and 
        self.legalLabels.
        """
        "*** YOUR CODE HERE ***"
        P = util.Counter()
        for l in trainingLabels:
            P[l] += 1
        P.normalize()
        self.P = P
        
        # Initialize stuff
        counts = {}
        totals = {}
        for f in self.features:
            counts[f] = {0: util.Counter(), 1: util.Counter()}
            totals[f] = util.Counter()
                     
        # Calculate totals and counts
        for i, datum in enumerate(trainingData):
            y = trainingLabels[i]
            for f, value in datum.items():
                counts[f][value][y] += 1.0
                totals[f][y] += 1.0 
                
        bestConditionals = {}
        bestAccuracy = None
        # Evaluate each k, and use the one that yields the best accuracy
        for k in kgrid or [0.0]:
            correct = 0
            conditionals = {}            
            for f in self.features:
                conditionals[f] = {0: util.Counter(), 1: util.Counter()}
                
            # Run Laplace smoothing
            for f in self.features:
                for value in [0, 1]:
                    for y in self.legalLabels:
                        conditionals[f][value][y] = (counts[f][value][y] + k) / (totals[f][y] + k * 2)
                
            # Check the accuracy associated with this k
            self.conditionals = conditionals              
            guesses = self.classify(validationData)
            for i, guess in enumerate(guesses):
                correct += (validationLabels[i] == guess and 1.0 or 0.0)
            accuracy = correct / len(guesses)
            
            # Keep the best k so far
            if accuracy > bestAccuracy or bestAccuracy is None:
                bestAccuracy = accuracy
                bestConditionals = conditionals
                self.k = k
                
        self.conditionals = bestConditionals
                
    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        
        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses
            
    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.        
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
        
        To get the list of all possible features or labels, use self.features and 
        self.legalLabels.
        """
        logJoint = util.Counter()
        evidence = datum.items()
        "*** YOUR CODE HERE ***"
        for y in self.legalLabels:
            logJoint[y] = math.log(self.P[y])
            for f in self.conditionals:
                prob = self.conditionals[f][datum[f]][y]
                logJoint[y] += (prob and math.log(prob) or 0.0)

        return logJoint
    
    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                        P(feature = 1 | label1) / P(feature = 1 | label2) 
        
        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        "*** YOUR CODE HERE ***"        
        featuresOdds = []
        for f in self.features:
            top = self.conditionals[f][1][label1]
            bottom = self.conditionals[f][1][label2]
            ratio = top / bottom
            featuresOdds.append((f, ratio))
            
        featuresOdds = [f for f, odds in sorted(featuresOdds, key=lambda t: -t[1])[:100]]

        return featuresOdds
        

        
            
