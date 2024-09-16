# Amazon-ML-Challenge
This repository contains our solutions for the Amazon ML Challenge 2024, which achieved a F1-score of `0.661` and secured a Top-30 finish (Public Leaderboard) among over 2000 participating teams.

### Problem Statement

Given the images of online products on Amazon with various measurements of physical quantities (e.g., height, width, weight) specified, the task was to **extract the numerical values corresponding to the physical quantities given as input**.

### Solution

- Our approach for extracting numerical values from images uses the state-of-the-art [PaliGemma-3B](https://huggingface.co/google/paligemma-3b-ft-docvqa-448) model by Google.
- We fine-tune the model on a sampled dataset of 40k image-value pairs from the train split, and report the results on the held-out test split.
- Our approach requires a 24GB GPU VRAM.
