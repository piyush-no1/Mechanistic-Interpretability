# Evaluating Latent Feature Drift Across Domain Shifts using Sparse Autoencoders

[cite_start]This repository contains the complete, reproducible pipeline for analyzing how a language model's internal latent representations shift during domain-specific fine-tuning[cite: 4, 31]. [cite_start]This project was developed as part of the **AIMS DTU Research Intern 2026** module in Mechanistic Interpretability[cite: 1, 2].

## Project Overview
[cite_start]Language model internal layers represent features densely, making individual dimensions difficult to interpret[cite: 5, 6]. [cite_start]By training Sparse Autoencoders (SAEs), we disentangle these dense representations into highly interpretable, sparse latent features[cite: 9, 10, 11]. 

[cite_start]This project explores a core question: **Does fine-tuning on a narrow task only alter domain-specific circuits, or does it quietly warp unrelated general features as well?** [cite: 21] 

[cite_start]We observe this phenomenon by tracking features inside a middle layer of **Pythia-160M** before and after fine-tuning it exclusively on a Python code corpus[cite: 4, 15, 16].

## Methodology
To isolate true feature drift from random coordinate rotations introduced by stochastic dictionary learning initialization, this pipeline uses an anchored training approach:
1. [cite_start]**Baseline SAE:** Trained on standard residual stream activations from a middle layer of base Pythia-160M[cite: 15].
2. [cite_start]**Domain Adaptation:** The base model is fine-tuned via LoRA on a targeted Python code dataset[cite: 16].
3. [cite_start]**Anchored Fine-Tuned SAE:** A second SAE is initialized directly with the weights of the baseline SAE ($W_{\text{enc}}$, $W_{\text{dec}}$, and biases) and trained with a conservative learning rate on the fine-tuned model's activations[cite: 19]. This anchors the basis coordinates, isolating true semantic mutation.

## Repository Structure
* `model_info.py` - Tells about the structure to the Pythia-160M.
* [cite_start]`fine_tune.py` - Sets up the target model domain adaptation pipeline[cite: 16].
* [cite_start]`cache_activations.py` / `cache_ft_activations.py` - Collects internal representations from the target layer[cite: 15, 19].
* [cite_start]`train_sae_v3.py` - Logic for training the baseline Sparse Autoencoder[cite: 15].
* [cite_start]`train_sae_ft.py` - Script for training the fine-tuned SAE using weight-anchoring constraints[cite: 19].
* [cite_start]`compare_saes.py` - Computes global alignment metrics and outputs similarity distributions[cite: 27].
* `probe_features.py` - Qualitatively probes the response profiles of individual tokens to check for feature specialization.

## Artifacts & Model Weights
[cite_start]In compliance with assignment specs, raw activation caches and heavy tensor checkpoints are hosted externally:
* **Baseline & Fine-Tuned SAE Checkpoints:** [INSERT YOUR GOOGLE DRIVE LINK HERE]
* **Activation Arrays:** [INSERT YOUR GOOGLE DRIVE LINK HERE]

## Key Findings (Teaser)
Our qualitative tracking validates that the coordinate space remains stable for general prose tokens while undergoing structural mutations on target programming keywords. For example, generic text tracking features (e.g., `#1604`) are suppressed on code syntax in favor of newly specialized coding structures (e.g., `#5575` and `#7038`).
