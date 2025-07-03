# Impression Network Analysis

This project analyzes an impression network of students based on data provided in an Excel sheet. It solves three tasks:

---

## Question 1: Find the Top Leader

- Simulates a random walk on the impression network with teleportation.
- The node (student) visited the most times is considered the most influential or "top leader."

---

## Question 2: Predict Missing Links

- Uses a matrix-based method to recommend new edges (connections) that might exist but were not observed.
- Helps identify relationships that are likely missing from the network.

---

## Question 3: Check Data Normality

- Analyzes the distribution of how many times each node appears in other students' impressions (in-links).
- Compares the distribution to an ideal bell curve.
- Identifies students with unusual impression patterns.

---

## How to Run

1. Make sure the data file `Impression network.xlsx` is in the project directory.

2. Install required Python libraries:
   ```bash
   pip install pandas numpy
3. Run the script:
   ```bash
   python project2.py
## Author

Chamala Venkata Rithish Reddy (2023CSB1113)
