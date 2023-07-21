# Thesis_Repo
Master's thesis implementation 
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


# Distillation between BERT and DistilBERT with Discrete Key-Value Bottleneck

This repository contains code for knowledge distillation from a teacher model (BERT with Discrete Key-Value Bottleneck) to a student model (DistilBERT with Discrete Key-Value Bottleneck) using the GLUE benchmark datasets. The distillation process aims to transfer knowledge from the more complex teacher model to the more compact and faster student model.

## Models

- `BERTwithBottleNeck`: BERT model with a discrete key-value bottleneck.
- `DistilBERTwithBottleNeck`: DistilBERT model with a discrete key-value bottleneck.

## Data Preprocessing

We use the `CustomDataset` class to preprocess and create custom datasets from the GLUE benchmark datasets for training, validation, and testing.

## Optimization

We use the AdamW optimizer to update the parameters of the student model during training.

## Distillation Loss

We calculate the distillation loss, which is the Kullback-Leibler (KL) divergence between the teacher and student model logits, scaled by a temperature factor. This loss guides the student model towards the teacher's predictions.

## Training Loop

We train the student model using the distillation loss and evaluate its performance on the validation set.

## Metrics

We calculate metrics such as F1 Score, Matthews Correlation Coefficient (MCC), and accuracy to evaluate the performance of the student model during training and validation.

## How to Use

To use this code for distillation between BERT and DistilBERT with Discrete Key-Value Bottleneck, follow these steps:

1. Install the required libraries: Make sure you have installed the necessary libraries, including Transformers and other dependencies.
2. Prepare Data: Load the GLUE benchmark datasets (MNLI, RTE, CoLA, MRPC, QNLI, SST2, STSB, QQP) and create custom datasets using the `CustomDataset` class.
3. Train the Student Model: Run the training loop to distill knowledge from the teacher to the student model. Adjust hyperparameters such as batch size, learning rate, and number of epochs as needed.
4. Evaluate the Student Model: Evaluate the performance of the student model on the validation set using F1 Score, MCC, and accuracy metrics.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is based on knowledge distillation techniques and makes use of the Transformers library. Special thanks to the authors and maintainers of these libraries for providing the tools to facilitate research in natural language processing.

