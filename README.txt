CMSC 191: Special Topics: Fundamentals of Neural Computing
A Neural Network Approach to Predicting the Approval or Rejection of a Credit Card Application
Author: Cutines, Adrian Carl B.

WHAT THIS IS
A one-hidden-layer feedforward neural network with logistic units, trained with the course's
batch backpropagation program (adaptive learning rates + momentum, QB64) to predict whether a
credit card application is rejected. Python (test.py) prepares the data and scores the results.

DATA
Kaggle dataset "Credit card details" (rohitudageri/credit-card-details): two files,
  - the INPUTS file (applicant attributes)      -> file name must contain "INPUTS"
  - the OUTPUT file (Ind_ID, label)             -> file name must contain "OUTPUT"
Label meaning (from the dataset description): 0 = application approved, 1 = application rejected.
1,548 applications, 175 rejected (11.3%). The rejected class is the "positive class" in the paper.

FILES IN THIS FOLDER
  backprop.bas       The course backpropagation program (QB64), extended as described in Section 4.3.
  test.py            Builds PROBLEM (preprocessing + grouped train/validation/test split) and scores results.
  network.txt        Run settings (one value per line, see below).
  PROBLEM            Generated data file (rebuilt by "python test.py"; the copy here is for seed 0).
  <INPUTS csv>       Data file with applicant attributes.
  <OUTPUT csv>       Data file with the labels.
  results_log.csv    Created by "python test.py score": one row per run, every number used in the paper.
  curve_*.png        Created by "python test.py score": training/validation error per epoch for each run.
  backprop.exe       Created by compiling backprop.bas (not needed to read the code).
  predictions.csv, trainlog.csv   Created by backprop.exe for the latest run.

REQUIREMENTS
  - Python 3 with: pip install numpy pandas matplotlib
  - QB64 (to compile backprop.bas)
Keep ALL files above in ONE folder; the programs look for each other in the current folder.

ONE-TIME SETUP
  1. Open backprop.bas in QB64 and choose  Run > Make EXE Only.
     This creates backprop.exe in the same folder.

HOW TO RUN ONE EXPERIMENT (repeat for every run in the table below)
  1. Put that run's values in network.txt (one value per line).
  2. python test.py              Builds PROBLEM from the CSV files (needed whenever the seed changes).
  3. Double-click backprop.exe   Wait for "DONE", then press Enter to close.
  4. python test.py score        Prints the confusion matrices, accuracy, precision, recall, F1 and AUC,
                                 appends a row to results_log.csv and saves curve_<settings>.png.

network.txt (11 lines, top to bottom)
  1 hidden nodes            2 kappa     3 phi     4 theta     5 mu
  6 jump connections (0/1)  7 maximum epochs      8 patience (epochs without improvement before stopping)
  9 seed (changes the split and the initial weights)
  10 split type (0 = repeated applicants kept together, 1 = plain random split)
  11 restore best-validation weights (1 = yes, 0 = keep the final weights)

THE SEVEN RUNS IN THE PAPER (network.txt, top to bottom)
  Run  hidden kappa phi  theta mu  jump epochs patience seed split restore
  A    16     0.1   0.7  0.7   0.8 0    2000   150      0    0     1
  B    4      0.1   0.7  0.7   0.8 0    2000   150      0    0     1
  C1   4      0.1   0.7  0.7   0.8 0    300    300      0    0     0
  C2   4      0.1   0.7  0.7   0.8 0    300    300      1    0     0
  C3   4      0.1   0.7  0.7   0.8 0    300    300      2    0     0
  C4   4      0.1   0.7  0.7   0.8 0    300    300      3    0     0
  C5   4      0.1   0.7  0.7   0.8 0    300    300      4    0     0

RESULTS REPORTED IN THE PAPER
  Runs C1-C5 (4 hidden nodes, 300 epochs, final weights, five different splits):
  mean test AUC 0.602 (SD 0.061); test AUC above 0.5 in every run (0.528 to 0.660).
  Accuracy, precision, recall and F1 depend strongly on the decision threshold and are less reliable
  than the AUC with only 26-27 positive cases per test set. Small differences are possible if you
  rerun on another machine or QB64 version.

NOTES
  - The data contain repeated applicants (same person under different Ind_ID values). Rows with the
    same "person key" (nine stable attributes, see Section 4.2) are always placed in the same split.
    This key is a heuristic.
  - Medians, modes and minimum/maximum values used for filling and scaling come from the training rows only.
  - Output of a run overwrites predictions.csv, trainlog.csv and PROBLEM; results_log.csv keeps every run
    (re-running the same settings replaces its row).
  - Jump connections are implemented in backprop.bas but were not used in the runs reported in the paper.
