# Deep Learning Applications – Laboratory Exercises

## Overview

This repository collects three laboratory sessions exploring different aspects of deep learning through practical experiments. The focus is not on reaching state-of-the-art performance, but on **understanding fundamental principles, experimenting with architectures, and analyzing model behaviors** across diverse tasks.

The labs progress from basic neural networks to transformers and robustness techniques, covering both **vision** and **language** domains. Each session introduces key methods and provides hands-on practice with modern frameworks such as **PyTorch** and **HuggingFace**.

---

## Laboratory 1 – Neural Networks and Explainability

The first lab introduces foundational deep learning models, starting with **Multilayer Perceptrons (MLPs)** and extending to **Convolutional Neural Networks (CNNs)**.
Key themes include:

* Comparing model capacity across small, medium, and large MLPs.
* Understanding the benefits of **residual connections** in both MLPs and CNNs.
* Training and evaluating CNNs of increasing depth, including **ResNet**-style architectures.
* Exploring **fine-tuning** by transferring models from CIFAR-10 to CIFAR-100.
* Applying **explainability techniques** such as Class Activation Maps (CAMs) to visualize model decision regions.

---

## Laboratory 3 – Transformers and Vision-Language Models

The third lab shifts the focus to **Transformers**, with applications in natural language processing and vision-language tasks.
The exercises cover:

* Building a **sentiment analysis** pipeline with DistilBERT on the Rotten Tomatoes dataset.
* Establishing a **baseline with frozen embeddings** and a simple classifier.
* **Fine-tuning** DistilBERT for improved sequence classification.
* Experimenting with **CLIP**, a vision-language model capable of zero-shot image classification.
* Applying **parameter-efficient fine-tuning (LoRA)** to adapt CLIP to specific datasets.

This lab emphasizes both **practical skills with HuggingFace tools** and a conceptual understanding of fine-tuning strategies for large pretrained models.

---

## Laboratory 4 – Robustness and Out-of-Distribution Detection

The fourth lab investigates the **robustness of deep learning models** and their ability to detect anomalous data.
Main topics include:

* Implementing **Out-of-Distribution (OOD) detection** using classifier confidence and autoencoder reconstruction error.
* Exploring **adversarial attacks** with the Fast Gradient Sign Method (FGSM), generating imperceptible perturbations that mislead models.
* Training models with **adversarial examples** to study the trade-off between adversarial robustness and OOD detection.
* Experimenting with **targeted attacks**, forcing misclassifications toward specific chosen labels.

This lab highlights critical challenges in deploying deep learning systems in real-world, safety-sensitive contexts.

---

## Repository Structure

Each laboratory follows a similar structure:

* **src/** – source code, including model definitions, training routines, and utilities.
* **models/** – saved models.
* **images/** – visualizations and examples.
* **data/** – datasets or dataset loaders (where applicable).

---

## Disclaimer

This work was completed as part of academic laboratory exercises.
Artificial Intelligence (AI) tools were used to provide **assistance in writing documentation** and to offer **suggestions for code structuring and clarity**. All experiments, implementations, and results were carried out by the authors.

These laboratory exercises were carried out in collaboration and discussion with Alessandra Spinaci, which may result in some similarities in the analytical workflows. However, each participant independently developed their own interpretations, conclusions, and comments throughout the exercises.