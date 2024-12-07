# bio_batch_BO

This repository contains the code used in the batch BO component of the AI-cyborg cardiac organoid maturation project conducted by Prof Na Li and Prof. Jia Liu's labs at Harvard University. 

## System requirements 

The code requires installation of the Anaconda package. All required packages are listed in the environment.yaml file. The software has been tested on a 13-inch, 2020 Macbook Pro with 2 GhZ Quad-Core Intel Core i5 processor, 16GB RAM. There are no nonstandard hardware required.

# Installation Guide

Follow these steps to set up and run the project on your local machine. The typical time required for installation is about 5 to 10 minutes.

---

## Prerequisites

Make sure you have the following tools installed (skip if you have installed these):

1. **Conda**
   - Download and install Anaonda from:
     - [Anaconda](https://www.anaconda.com/products/distribution)
   - Verify the installation by running:
     ```bash
     conda --version
     ```

2. **Git**
   - Download and install Git from [git-scm.com](https://git-scm.com/).
   - Verify the installation by running:
     ```bash
     git --version
     ```

---
## System-specific instructions

If on Mac/Linux, perform the following steps in the terminal.
If on Windows, open up the Anaconda Powershell Prompt (can find this by searching in the Windows search bar), and then perform the following steps in the Anaconda Powershell prompt.

## Step 1: Clone the Repository

Clone this repository to your local machine using Git:

```bash
git clone https://github.com/CoNG-harvard/bio_batch_BO.git
cd bio_batch_BO
```

## Step 2: Create and then activate conda environment

```
conda env create -f environment.yml
conda activate bio_batch_BO
```

## Step 3: Try out demo notebook

For a demonstration of how to run the code, run the command 

```
jupyter notebook
```
and then open up the notebook demo.ipynb. 

## Details on demo notebook
The demo notebook loads in simulated data, which comprises 3 arrays: xp, yp, and pseudotime_current_devices. xp comprises the pseudotime and action in the existing dataset, yp comprises the change in pseudotime measured at the corresponding pseudotimes and actions, while pseudotime_current_devices contains the pseudotime of the existing current batch of devices for which we wish to sample actions. The expected output is a new set of actions to sample. The expected runtime on the demo is about 1 minute. 

## Instructions for use

To run the software on your own data, first preprocess the data and create a new python .npz file with the xp, yp and pseudotime_current_devices for your problem. Then, run the demo notebook to generate a new batch of actions.




Acknowledgement: this code is adapted and modified from [Gaussian Max-Value Entropy Search for Multi-Agent Bayesian Optimization](https://github.com/mahaitongdae/dbo) by [Haitong Ma](https://github.com/mahaitongdae), which in turn is modified from [Distributed Bayesian Optimization for Multi-Agent Systems](https://github.com/FilipKlaesson/dbo) created by [Filip Klaesson](https://github.com/FilipKlaesson). 

