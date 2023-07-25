# Bottleneck Distillation Model with Key Carrying (Teacher to Student)

This repository contains an implementation of a bottleneck distillation model, where knowledge is transferred from a larger teacher model to a smaller student model. The key innovation in this model is the use of discrete keys that are carried from the teacher model to the student model during training. This approach allows for efficient knowledge transfer while reducing the computational cost of the student model.

## Model Overview

The bottleneck distillation model consists of two main components:

1. Teacher Model: The teacher model is a BERT-based model with a bottleneck layer that utilizes discrete keys. The discrete keys are maintained as an exponential moving average (EMA) for better stability during training.

2. Student Model: The student model is a DistilBERT-based model with a similar bottleneck layer as the teacher model. The goal is to train this smaller model to mimic the knowledge of the larger teacher model.

## Key Steps of Training

The training process involves the following key steps:

1. Load the teacher and student models with their respective architectures.

2. Transfer Weights from Teacher to Student: The weights of the teacher model are transferred to the student model to initialize its parameters.

3. Create Data Loaders: The training, validation, and testing datasets are loaded and transformed into data loaders for efficient batch processing.

4. Define the Distillation Loss: During training, the student model is optimized to minimize the distillation loss, which measures the similarity between the teacher and student model predictions.

5. Carry Over Discrete Keys: In the bottleneck layer, discrete keys from the teacher model are copied to the student model for effective knowledge transfer. This key carrying mechanism ensures that the student model can leverage the knowledge encoded in the teacher model's discrete keys.

6. Training and Evaluation: The student model is trained using the distillation loss and evaluated on the validation set. The process is repeated for multiple epochs to fine-tune the student model and achieve better performance.

7. Test the Student Model: After training, the student model's performance is evaluated on the test set to assess its generalization ability and knowledge transfer from the teacher model.

## How to Use this Repository

To use the bottleneck distillation model with key carrying, follow these steps:

1. Install the required dependencies by running `pip install -r requirements.txt`.

2. Prepare your dataset in the GLUE format and load it using the provided `CustomDataset` class.

3. Modify the hyperparameters (e.g., batch size, learning rate, temperature) as needed.

4. Execute the training script to train the student model with knowledge transferred from the teacher model.

5. Evaluate the student model on the test set and analyze its performance metrics.

## Additional Notes

- The implementation is based on PyTorch and leverages popular transformer models such as BERT and DistilBERT.

- Ensure that you have the appropriate GPU capabilities to accelerate training if using large datasets.

- For more details on the architecture and training process, refer to the source code and comments in the Python files.

- Feel free to experiment with different hyperparameters, architectures, and distillation strategies to improve the model's performance.

Happy experimenting with the bottleneck distillation model with key carrying! If you have any questions or suggestions, please don't hesitate to reach out.
