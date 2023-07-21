# DiscreteKeyValueBottleneck for Knowledge Distillation

This repository contains code for the DiscreteKeyValueBottleneck model used in knowledge distillation. The model aims to reduce the complexity of a teacher model for knowledge transfer to a smaller and faster student model.

## Model Architecture

The `DiscreteKeyValueBottleneck` class is the core component of the knowledge distillation process. It takes a teacher model (BERT Base) as input and generates a student model (DistilBERT with Discrete Key-Value Bottleneck) using a VectorQuantize module. The key-value bottleneck is designed to compress the input embeddings and optimize the keys using an exponential moving average (EMA) algorithm.

## Data Preprocessing

This repository includes code for loading and preprocessing the MNIST dataset. The code separates the dataset into individual classes for better control during training and evaluation.

## Training and Optimization

The knowledge distillation process is performed by training the student model with a discrete key-value bottleneck. The model's parameters are optimized using the Adam optimizer, and the loss function is based on the Kullback-Leibler (KL) divergence between the teacher and student model logits.

## Key Optimization

The `discrete_key_init` function optimizes the keys of the discrete key-value bottleneck during the knowledge distillation process. It involves updating the embeddings of the student model's codebooks with a weighted sum of the closest embeddings from the teacher model.

## Evaluation

The model's performance is evaluated on the validation set using metrics such as loss and accuracy. The best-performing model is saved and used for final testing on the test dataset.

## How to Use

1. Install the required libraries: Make sure you have installed PyTorch, torchvision, einops, and tqdm.
2. Prepare Data: Load the MNIST dataset and preprocess it, splitting it into training, validation, and test sets.
3. Define Model: Create the teacher model (BERT Base) and the student model (DistilBERT with Discrete Key-Value Bottleneck) using the provided classes.
4. Training and Distillation: Train the student model using knowledge distillation from the teacher model, and perform key optimization with the `discrete_key_init` function.
5. Evaluation: Evaluate the trained student model on the validation set and select the best-performing model for testing.
6. Test: Test the final model on the test dataset to measure its accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project uses PyTorch, einops, and torchvision libraries. Special thanks to the authors and maintainers of these libraries for their valuable contributions to the deep learning community.
