# Demo-Repository-of-the-E2E-Portfolio-Strategy
  This demo repository showcases the core ideas from the paper “An End-to-End Deep Learning Framework for Portfolio Optimization with Stop-Loss Orders.”
**Environment Dependencies:**
  This demo requires Python (3.7+ recommended) with the following packages installed: the built-in os and time modules; PyTorch (the torch core library, including torch.nn and torch.distributions); NumPy; Joblib; CVXOPT (for matrix and solvers); and Matplotlib (specifically matplotlib.pyplot).
**Data Description:** 
  All required data files are located in the data_a/ folder, containing raw and processed DJIA dataset files; 
  ph.csv: Daily highest prices; 
  pl.csv: Daily lowest prices; 
  pc.csv: Daily closing prices; 
  vol.csv: Daily trading volume; 
  data_s.pkl: Technical indicators (model input features); 
  rc_out.pkl: Closing returns; 
  rh_out.pkl: Highest returns; 
  rl_out.pkl: Low returns.
**Quick Start:**
  Simply run the main script Code_Sample.py—it will automatically load the prepared DJIA data from the data_a/ folder, execute the model’s end-to-end pipeline, and print out key empirical results from the paper without any additional setup.
**Note:** 
  If you wish to regenerate all intermediate files from scratch, please refer to the data-preprocessing steps in the paper’s appendix or add a preprocessing module directly into Code_Sample.py.
