# Files in the janus repository
The file janus.py is a python script to numerically integrate Janus oscillator networks using SciPy's Complex-valued Variable-coefficient Ordinary Differential Equation solver.  The file plot.nb is a Mathematica notebook which can be used to plot results.  The folder data contains the output generated by the examples below.
# System requirements
This python code has been run with anaconda2 and anaconda3, either of which can be downloaded from the Anaconda website: https://www.anaconda.com/download/#macos.  It requires packages numpy, argparse, scipy, and progressbar. Create an environment with these packages with `conda create -n janus_env scipy numpy argparse progressbar` and activate it with `source activate janus_env`.  The Mathematica code has been run with Mathematica 11.1.1.0.
# Usage
Running the terminal command `./janus.py -h` will give a usage message with command line argument descriptions, using the argparse package.
```
usage: janus.py [-h] --filebase FILEBASE [--sweep {0,1}] [--output {0,1}]
                [--number NUMBER] [--time TIME] [--atime ATIME]
                [--rtime RTIME] [--dt DT] [--avgcount AVGCOUNT]
                [--threshold THRESHOLD] [--beta0 BETA0] [--beta1 BETA1]
                [--sigma0 SIGMA0] [--sigma1 SIGMA1] [--delta0 DELTA0]
                [--delta1 DELTA1] [--omega1 OMEGA1] [--omega2 OMEGA2]
                [--iseed ISEED] [--hseed HSEED] [--dsigma DSIGMA]
                [--rthreshold RTHRS] [--pthreshold PTHRS]
                [--neighbors NEIGHBORS] [--links NUMLINKS] [--dimension {1,2}]

Numerical integration of networks of Janus oscillators.

optional arguments:
  -h, --help            show this help message and exit
  --filebase FILEBASE   Base string for file output
  --sweep {0,1}         Flag to run a branch sweep; 0 for single runs and 1
                        for sweeps. Default 0.
  --output {0,1}        Flag for output style; 0 for abbreviate data and 1 for
                        full data. Default 0.
  --number NUMBER       Number of Janus oscillators. Default 50.
  --time TIME           Total integration time. Detault 10000.
  --atime ATIME         Time to adiabatically change coupling from sigma0 to
                        sigma. Default 0.
  --rtime RTIME         Time to start averaging order parameter. Default 9000.
  --dt DT               Time between outputs. Default 0.1.
  --avgcount AVGCOUNT   Number of timesteps to average over for cluster
                        counting. Default 10.
  --threshold THRESHOLD
                        Frequency difference threshold for cluster counting.
                        Default 0.01.
  --beta0 BETA0         Initial internal coupling constant. Default 0.25.
  --beta1 BETA1         Final internal coupling constant. Default 0.25.
  --sigma0 SIGMA0       Initial external coupling constant. Default 0.4.
  --sigma1 SIGMA1       Final external coupling constant. Default 0.4.
  --delta0 DELTA0       Initial frequency heterogeneity. Default 0.0.
  --delta1 DELTA1       Final frequency heterogeneity. Default 0.0.
  --omega1 OMEGA1       Natural frequency 1. Default 0.5.
  --omega2 OMEGA2       Natural frequency 2. Default -0.5.
  --iseed ISEED         Initial condition random seed. Default 5.
  --hseed HSEED         Heterogeneity profile random seed. Default 1.
  --dsigma DSIGMA       Coupling strength sweep step size. Default 0.002.
  --rthreshold RTHRS    Threshold change in order parameter to stop branch
                        sweep. Default 0.05.
  --pthreshold PTHRS    Threshold change in num locked to stop branch sweep.
                        Default 5.0.
  --neighbors NEIGHBORS
                        Number of neighbors. Default 1.
  --links NUMLINKS      Number of random small-world links. Default 0.
  --dimension {1,2}     Network structure dimension. Default 1. --dimension 2
                        has not been tested for all options.

```
For backwards-compatibility with previous command line argument structure, running the terminal command `python janus.py` will produce the different usage message below.  Either style of command line arguments will function.  In the new style, the `--sweep 0` argument corresponds to usage 1, and the `--sweep 1` argument corresponds to usage 2. In the argparse style, the first example can be run with `./janus.py --filebase data/random/random`, and the second can be run with `./janus.py --time 10000 --atime 4000 --rtime 5000 --sweep 1 --filebase data/branches/chimera`.

```
Usage will depend on number of command line argument  

usage 1: `python explosive.py [N] [t1] [t2] [t3] [dt] [avgcount] [thrs] [beta0] [beta] [sigma0] [sigma] [delta0] [delta] [seed] [seed2] [filebase] [output]`  
Used for single simulation runs with adiabatic parameter changes  
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
delta0 is initial frequency heterogeneity  
delta is final frequency heterogeneity  
seed is random seed for initial condition (if filebaseic.dat does not exist, otherwise initial conditions from file are used)  
seed2 is random seed for heterogeneity  
filebase is the output file string base; output files are filebaseout.dat and filebaseorder.dat  
output is 1 to output time data and 0 for shorter output  
Example 1) `python janus.py 50 10000 0 9000 0.1 10 1e-2 0.25 0.25 0.4 0.4 0.0 0.0 5 1 data/random/random 1`  
This example will likely produce a chimera state.  The final state data/random/randomfs.dat can be copied to chimeraic.dat to save as an initial condition. Run the following command to copy the file for Example 2 below.  
`mkdir -p data/branches && cp data/random/randomfs.dat data/branches/chimeraic.dat`

usage 2: `python explosive.py [N] [t1] [t2] [t3] [dt] [avgcount] [thrs] [sigma0] [beta] [delta] [dsigma] [rthrs] [pthrs] [seed2] [filebase]`  
Used to adiabatically sweep out a solution branch  
N is the number of oscillators  
t1 is the total integration time  
t2 is the time to adiabatically change coupling from sigma0 to sigma  
t3 is the time to start averaging r  
dt is the time between outputs  
avgcount is the number of timesteps to average over for cluster counting  
thrs is the frequency difference threshold for cluster counting  
sigma0 is the initial external coupling constant  
beta is the internal coupling constant  
delta is frequency heterogeneity  
dsigma is the coupling strength step  
rthrs is the threshold change in order parameter to stop branch sweep  
pthrs is the threshold change in num locked to stop branch sweep  
seed2 is random seed for the heterogeneity profile  
filebase is the output file string base; output files are filebaseout.dat and filebaseorder.dat  
Example 2) `python janus.py 50 10000 4000 5000 0.1 10 1e-2 0.4 0.25 0.0 0.002 0.05 5.0 1 data/branches/chimera`  
This will sweep out the coupling constant for the chimera initial condition.  
```
___
# Output files
Usage 1  
The program always appends a summary line of statistics about the simulation to the file filebaseorder.dat, which contains the coupling constant sigma, the heterogeneity strength delta, the mean order parameter r, and the mean number of phase-locked oscillators Nlocked, where the means are calculated from time t2, to t1, separated by spaces. The program also always saves the final state of the system to filebasefs.dat (which can be copied to another filebaseic.dat to use as initial conditions elsewhere). If the output flag is set to 1, the program also creates a file filebaseout.dat that contains more detailed data.  The first line of filebaseout.dat contains the command line that ran the program.  The second line contains the natural frequencies of each phase oscillator in each Janus oscillator.  The third line contains the instantaneous order parameter at each timestep.  The next 2N lines contains the phases of each phase oscillator in each Janus oscillator for each timestep. The next N lines contains the adjacency matrix of the network of phase oscillators.  If the output flag is high, the program also creates a file filebaseclusters.dat that contains information about which oscillators were deemed instantaneously phase-locked.  Each line corresponds to every avgcount timesteps.  The first N entries of each line are 0 for unlocked oscillators and 1 for locked oscillators.  The remaining entries on these lines are the average instantaneous phase velocity for each of the oscillators that was deemed locked.  

Usage 2  
The program creates a filebasesweep.dat, which contains the summaries in the form of the filebaseorder.dat from usage 1 for each sigma in the branch sweep.  The program also creates a filebasecrits.dat which contains the smallest and largest values of the coupling constant sigma for which the branch exists.
