# janus
Code to simulate rings of Janus oscillators
____
usage: python janus.py [N] [t1] [t2] [t3] [dt] [avgcount] [thrs] [sigma0] [sigma] [dim] [k] [numlinks] [delta0] [delta] [seed] [seed2] [output] [filebase]

  N is the number of oscillators

  t1 is the total integration time

  t2 is the time to adiabatically change coupling from sigma0 to sigma

  t3 is the time to start averaging r

  dt is the time between outputs

  avgcount is the number of timesteps to average over for cluster counting

  thrs is the frequency difference threshold for cluster counting

  beta0 is the initial internal coupling constant

  beta is the final internal coupling constant

  sigma0 is the initial external coupling constant

  sigma is the final external coupling constant

  dim is network dimension, 1 for ring, 2 for grid

  k is number of neighbors to include in graph 

  numlinks is expected number of random links to add (positive) or remove (negative) in graph 

  delta0 is initial frequency heterogeneity 

  delta is final frequency heterogeneity 

  seed is random seed for initial condition

  seed2 is random seed for heterogeneity

  output is 1 to output time data

  filebase is the output file string base; output files are filebaseout.dat and filebaseorder.dat
___
Example I: python janus.py 50 1000 0.1 0.1 0.1 10 1e-2 0.25 0.25 0.35 0.35 1 1 0 0.0 0.0 1 1 data/test/test 1
___
Example II: python janus.py 24 1000 0.1 0.1 0.1 10 1e-2 0.3 0.3 0.21 0.21 2 1 0 0.0 0.0 1 1 data/test/test 1
