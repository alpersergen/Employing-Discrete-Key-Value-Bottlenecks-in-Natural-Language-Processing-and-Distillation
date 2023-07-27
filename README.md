# Knowledge Distillation with DiscreteKeyValueBottleneck

This repository contains implementations of various knowledge distillation models using the DiscreteKeyValueBottleneck for compressing teacher models and transferring knowledge to smaller student models. Below is a summary of each model and its corresponding proof-of-concept markdown file:

## Proof of Concept Models

1. [Proof of Concept](https://github.com/alpersergen/Thesis_Repo/blob/main/Proof_of_Concept/Readme.md): This markdown file showcases the initial proof of concept for the DiscreteKeyValueBottleneck model. It demonstrates how the model can be used to compress the embeddings from a teacher model (BERT Base) and optimize the keys using an exponential moving average (EMA) algorithm.

2. [Distillation with Key Carried](https://github.com/alpersergen/Thesis_Repo/blob/main/Bottleneck_Distillation_Keys_Carried/Readme.md): This markdown file presents a bottleneck distillation model where keys are carried over from the teacher(BERT with Discrete Key Value Bottleneck) to the student model(DistilBERT with Discrete Key-Value Bottleneck). It includes code for loading and preprocessing the MNIST dataset, training the student model, and optimizing the keys during the knowledge distillation process.

3. [Distillation with New Keys ](https://github.com/alpersergen/Thesis_Repo/blob/main/Bottleneck_Distillation_With_New_Keys/Readme.md): In this markdown file, a bottleneck distillation model is demonstrated where keys are not carried over from the teacher to the student model. The teacher model (BERT with Discrete Key Value Bottleneck) is used to generate the student model (DistilBERT with Discrete Key-Value Bottleneck) without any key optimization.

4. [Distillation from Base BERT](https://github.com/alpersergen/Thesis_Repo/blob/main/Bottleneck_Only_Student_Network/Readme.md): This markdown file showcases a knowledge distillation process where a teacher model (BERT Base) transfers knowledge to a student model (DistilBERT with Discrete Key-Value Bottleneck) without any key carrying. The model's parameters are optimized using the Adam optimizer and the loss function based on the Kullback-Leibler (KL) divergence.

## How to Use

Each proof-of-concept markdown file provides detailed instructions on how to use the corresponding model and run the code. To get started, make sure you have the required libraries installed, preprocess the dataset, define the teacher and student models, and perform the knowledge distillation process.

## Acknowledgments

This project uses PyTorch, einops, and torchvision libraries. Special thanks to the authors and maintainers of these libraries for their valuable contributions to the deep learning community.

If you have any questions or suggestions, please don't hesitate to reach out.
