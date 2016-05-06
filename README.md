# Boston311

<b>I. Code Directory</b>
  1. <i>Reading in Data.ipynb</i>: reads the 311 data into a pandas data frame. Supports basic exploration of the entire set of 311 data.
  2. <i>Basic Analysis of Data.ipynb</i>: provides rudimentary visualization and analysis of closed requests generated in the year of 2015. Contrast and comparison between requests originating from call data and Citizens Connect App data is emphasized throughout.
  3. <i>Simulated Annealing.ipynb</i>: provides maximum likelihood estimations of the mixture model pa- rameters through simulated annealing. Includes convergence analysis and visualizations of hard clusterings of the data based on mixture parameter estimates. Contrast and comparison between requests originating from call data and Citizens Connect App data is emphasized throughout.
  4. <i>EM for MLE and MAP.ipynb</i>: provides maximum likelihood and maximum a posteriori estimates of the mixture model parameters through expectation maximization. Includes:
    1. performance testing on synthetic data
    2. model selection for the number of mixture components using Bayesian information criterion
    3. convergence analysis on real data
    3. hard clustering of the data based on MAP mixture parameter estimates
    4. cluster profile analysis
  
  Contrast and comparison between requests originating from call data and Citizens Connect App data is emphasized throughout.
  5. <i>Gibbs Sampler for GMM.ipynb<i>: provides Gibbs sampling from the posterior distribution of the mixture model. Including:
    1.convergence analysis
    2. hard clustering of the data based on posterior mean estimates of the mixture parameters
    3. visualization of the posterior predictive
    3. comparison of performance against basic MH sampling implemented in PyMC
    4. alternative 1-D model for response time as a mixture of exponentials
  
  Contrast and comparison between requests originating from call data and Citizens Connect App data is emphasized throughout.
  
  This notebook also ran the PyMC implementation of the Gaussian mixture model (not converged) and examined alternative modeling of response times only. 

<b>II. Poster Directory</b>
  1. <i>Boston311 poster.pdf</i>: poster with overview of the project including select results

<b>III. Write up Directory</b>
  2. <i>Boston311 paper.pdf</i>: summary, with details, of the methods and results as well as analysis.
