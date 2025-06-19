# NYCU-CS-Introduction-of-Artifical-Intelligence
本次專案為交大資工所大三必修-人工智慧導論，共有四份專案內容，分別為：路線最佳化、四子棋、CNN和DT比較、Multi-Armed Bandit

## Author：國立陽明交通大學 資訊管理與財務金融學系財務金融所碩一 313707043 翁智宏

This repository contains all assignments completed for the "Introduction to Artificial Intelligence" course (Spring 2025), including implementations for search algorithms, adversarial game agents, supervised learning models, and reinforcement learning experiments.

## 📁 Homework Overview

| Homework | Title | Topic | Due Date |
|---------|-------|-------|----------|
| HW1 | Route Finding | Search Algorithms (BFS, DFS, UCS, A*) | 2024/03/21 |
| HW2 | Connect Four AI | Minimax, Alpha-Beta, Custom Heuristic Agent | 2025/04/04 |
| HW3 | Supervised Learning | CNN + Decision Tree Image Classifier | 2025/04/21 |
| HW4 | Multi-Armed Bandit | Reinforcement Learning (ε-greedy, Non-stationary env) | 2025/05/13 |

---

## 🧭 HW1: Route Finding with Search Algorithms

Implement various search algorithms to find paths in real-world map data (Hsinchu City) from OpenStreetMap.

### 🔧 Features
- BFS / DFS / UCS / A* search implementations.
- Interactive visualization with `main.ipynb` (folium + Jupyter).
- Heuristic values provided in `heuristic_values.csv`.

### ✅ Evaluation
- BFS: 10%
- DFS: 10%
- UCS: 20%
- A*: 20%
- Testing (10 scenarios): scaled scoring
- Demo session: 40%

---

## 🎮 HW2: Connect Four Adversarial AI

Develop intelligent agents for the Connect Four game using adversarial search techniques.

### 🔧 Features
- Minimax Agent (Depth-4)
- Alpha-Beta Pruning Agent
- StrongAgent with custom heuristic function
- GUI-based game and CLI batch simulation
