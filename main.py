# -*- coding: utf-8 -*-
import plotting
import NaiveBayes
import Base_DT
import Best_DT
import Perceptron
import Base_MLP
import Best_MLP

##--------------------

NaiveBayes.run_naivebayes()

Base_DT.run()
Best_DT.run()

Perceptron.run_perceptron()

Base_MLP.run()
Best_MLP.run()