# AML Concept Drift & Incremental Learning Lab

This repository contains experiments for an Adaptive Machine Learning (AML) lab focused on Concept Drift and Incremental (Online) Learning. It stress-tests various streaming classifiers against both dynamically generated synthetic data and noisy real-world datasets.

## Project Overview

Standard batch-learning models fail when the underlying rules of the data change over time. This project demonstrates how online learning models adapt to sudden shifts in data distributions (Concept Drift). 

### Key Features
- **Synthetic Data Generation:** Simulates abrupt concept drift by dynamically flipping classification rules over continuous, asymmetric (Log-Normal) data streams.
- **Real-World Benchmarking:** Tests models against the famous `Elec2` dataset (Australian electricity market) to predict price fluctuations based on shifting seasonal and daily demands.
- **Custom Model Implementation:** Includes a custom Incremental Extreme Learning Machine (IELM) using Recursive Least Squares (RLS).
- **Automated Visualization:** Automatically generates matplotlib graphs for data distributions and rolling accuracy histories.

## 🛠️ Technologies & Models Used

**Libraries:** `numpy`, `matplotlib`, `pandas`, `river`

**Incremental Models Evaluated:**
1. **Gaussian Naive Bayes** (`river.naive_bayes.GaussianNB`)
2. **Hoeffding Tree** (`river.tree.HoeffdingTreeClassifier`)
3. **SGD Perceptron** (`river.linear_model.Perceptron`)
4. **Adaptive Random Forest** (`river.forest.ARFClassifier`) - *Includes built-in ADWIN drift detection!*
5. **Custom IELM** (Incremental ELM)
