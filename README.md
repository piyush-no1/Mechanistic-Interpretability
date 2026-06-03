# Evaluating Latent Feature Drift Across Domain Shifts using Sparse Autoencoders

This repository contains the complete, reproducible pipeline for analyzing how a language model's internal latent representations shift during domain-specific fine-tuning. This project was developed as part of the AIMS DTU Research Intern 2026 module in Mechanistic Interpretability.

## Project Overview
Language model internal layers represent features densely, making individual dimensions difficult to interpret. By training Sparse Autoencoders (SAEs), we disentangle these dense representations into highly interpretable, sparse latent features.

This project explores a core question: Does fine-tuning on a narrow task only alter domain-specific circuits, or does it quietly warp unrelated general features as well?

We observe this phenomenon by tracking features inside a middle layer of Pythia-160M before and after fine-tuning it exclusively on a Python code corpus.

## Methodology
To isolate true feature drift from random coordinate rotations introduced by stochastic dictionary learning initialization, this pipeline uses an anchored training approach:

* **Baseline SAE:** Trained on standard residual stream activations from a middle layer of base Pythia-160M.
* **Domain Adaptation:** The base model is fine-tuned via LoRA on a targeted Python code dataset.
* **Anchored Fine-Tuned SAE:** A second SAE is initialized directly with the weights of the baseline SAE ($W_{\text{enc}}$, $W_{\text{dec}}$, and biases) and trained with a conservative learning rate on the fine-tuned model's activations. This anchors the basis coordinates, isolating true semantic mutation.

## Repository Structure
* `model_info.py` - Explains the underlying layer architecture of the Pythia-160M model.
* `fine_tune.py` - Sets up the target model domain adaptation pipeline.
* `cache_activations.py` / `cache_ft_activations.py` - Collects intermediate activation states from the target layer.
* `train_sae_v3.py` - Logic for training the baseline Sparse Autoencoder.
* `train_sae_ft.py` - Script for training the fine-tuned SAE using weight-anchoring constraints.
* `compare_saes.py` - Computes global alignment metrics and outputs similarity distributions.
* `probe_features.py` - Qualitatively probes the response profiles of individual tokens to check for feature specialization.

## Artifacts & Model Weights
In compliance with assignment specs, raw activation caches and heavy tensor checkpoints are hosted externally:
* **Baseline & Fine-Tuned SAE Checkpoints:** [INSERT YOUR GOOGLE DRIVE LINK HERE]
* **Activation Arrays:** [INSERT YOUR GOOGLE DRIVE LINK HERE]

## Key Findings (Teaser)
Our qualitative tracking validates that the coordinate space remains stable for general prose tokens while undergoing structural mutations on target programming keywords. For example, generic text tracking features (e.g., #1604) are suppressed on code syntax in favor of newly specialized coding structures (e.g., #5575 and #7038).
