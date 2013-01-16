## CSE 250B Project 1

### Goal
* understand: logistic regression, gradient-based optimization, practicialities of classifier learning
* implement code for training logistic regression models with L2 regularization
* implement stochastic gradient descent (SGD)
* use a standard library package for L-BFGS
* reproduce results for logistic regression in Figure 3 of the paper "t-Logistic Regression" by Nan Ding and S.V.N. Vishwanathan (NIPS, 2010)

### Report Outline 
1. Introduction
2. Design and analysis of algorithms
3. Design of experiments
4. Results of experiments
5. Findings and lessons learned


### Figure 3
#### Caption
The test error rate of various algorithms on six datasets (left to right, top: Long-Servedio, Mease-Wyner, Mushroom; bottom: USPS-N, Adult, Web) with and without 10% label noise. All algorithms are initialized with θ = 0. The blue (light) bar denotes a clean dataset while the magenta (dark) bar are the results with label noise added. Also see Table 3 in the supplementary material.

#### Key points
* analysizes six datasets: Long-Servedio, Mease-Wyner, Mushroom, USPS-N, Adult, Web
* plot shows test error (%) with and without 10% label noise
* all algorithms initalized with theta=0
* only need to present results using logistic regression (not t-logistic or SVM)
