# HEC Master Thesis

This repository features the main paper as well as the replication package for my Master Thesis for the Major in International Finance, HEC Paris. 

I start by replicating the original results of: 
Bayer, C., Belomestny, D., Butkovsky, O., & Schoenmakers, J. (2024). 
A reproducing kernel Hilbert space approach to singular local stochastic volatility McKean–Vlasov models. 
Finance and Stochastics, 28(4), 1147-1178.

The code is available both in Python, with which I originally started, and in Matlab, which I continued with when speed started to become an issue with Python alone. 

I Then tried to improve upon their model by replacing the Ridge Regression with a Gaussian Process Regression and developping upper and lower bound for the precision of the resulting estimate of the common conditional expectation. 
