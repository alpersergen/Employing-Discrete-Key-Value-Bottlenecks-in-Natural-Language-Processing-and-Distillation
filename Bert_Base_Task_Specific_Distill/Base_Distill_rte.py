import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, glue_compute_metrics
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from tqdm import trange, tqdm
from einops import rearrange
import wandb
from datasets import load_dataset
from transformers import BertTokenizer
from einops import repeat

# Initialize Weights and Biases
wandb.login(key="a6d93dd680e09e7dddae91cf4cf664991f59025f")

wandb.init(
    # set the wandb project where this run will be logged
    project="knowledge-distillation",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-3,
    "architecture": "Bottleneck",
    "dataset": "Mnist-10",
    "epochs": 100,
    }
)
# Define your constants
MAX_LEN = 512
n_labels = 2
num_epochs = 5
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
LEARNING_RATE = 1e-5
LEARNING_RATE2= 2e-5
TEMPERATURE = 1.0

# Load dataset
dataset = load_dataset("glue", "rte")
train_dataset = dataset['train']
valid_dataset = dataset['validation']
test_dataset = dataset['validation']

# Define your tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define CustomDataset
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text1 = dataframe['sentence1']#dataframe.sentence
        self.text2 = dataframe['sentence2']
        self.targets = dataframe['label']
        self.max_len = max_len

    def __len__(self):
      return len(self.text1) 

    def __getitem__(self, index):
        text1 = str(self.text1[index])
        text1 = " ".join(text1.split())
        text2 = str(self.text2[index])
        text2 = " ".join(text2.split())

        inputs = self.tokenizer.encode_plus(
            text1,
            text2,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

from transformers import BertModel

# Create dataloaders
training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
validation_set = CustomDataset(valid_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

train_loader = DataLoader(training_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=VALID_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testing_set, batch_size=VALID_BATCH_SIZE, shuffle=True)

# Define teacher model

class BERTbase(nn.Module):
    def __init__(self,n_labels):
        super(BERTbase, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.n_labels = n_labels
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)


    def forward(self, ids, mask, token_type_ids):
        outputs = self.encoder(ids, attention_mask=mask, token_type_ids=token_type_ids)
        dropout_output = self.l2(outputs[1])
        logits = self.l3(dropout_output)

        return logits
from transformers import DistilBertModel
from torch import nn
from einops import rearrange
# Define your student model without bottleneck
class DistilBERTNoBottleNeck(nn.Module):

    def __init__(self, n_labels):
        super(DistilBERTNoBottleNeck, self).__init__()
        self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.n_labels = n_labels
        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)

    def forward(self, ids, mask):
        outputs = self.encoder(ids, attention_mask=mask)
        pooled_output = outputs[1]  # Use the pooled output from DistilBERT
        dropout_output = self.l2(pooled_output)
        logits = self.l3(dropout_output)

        return logits

# Rest of your code ...
import torch
from transformers import BertModel, BertConfig, DistilBertModel, DistilBertConfig
from torch import cuda
from tqdm import tqdm, trange

import pandas as pd
from transformers import BertTokenizer
from transformers import AdamW
import torch
device = 'cuda' if cuda.is_available() else 'cpu'

criterion = nn.BCEWithLogitsLoss()

def get_logits(model, ids, mask,token_type_ids):
  """
  Given a BERT model for classification and the couple of (input_ids) and (attention_mask),
  returns the logits corresponding to the prediction.
  """

  outputs = model(ids=ids,mask=mask,token_type_ids=token_type_ids)
  last_hidden_state = outputs[0]
  logits = model(ids=ids, mask=mask,token_type_ids=token_type_ids)

  # logits = outputs[:2].logits
  return last_hidden_state


def get_Slogits(model, ids, mask):
  """
  Given a BERT model for classification and the couple of (input_ids) and (attention_mask),
  returns the logits corresponding to the prediction.
  """

  outputs = model(ids=ids,mask=mask)
  last_hidden_state = outputs[0]

  return last_hidden_state
# Load the teacher and student models
teacher_model = BERTbase(n_labels = n_labels).to(device)
student_model = DistilBERTwithBottleNeck(n_labels = n_labels,pooling_type='mean',pool_before=False).to(device)

# Print the network architectures
print("Teacher Model:")
print(teacher_model)

print("\nStudent Model:")
print(student_model)

# Transfer weights from the teacher model to the student model
for teacher_name, teacher_param in teacher_model.named_parameters():
    if teacher_name not in student_model.state_dict():
        print(f"Ignoring {teacher_name} as it is not present in the student model")
        continue
    student_param = student_model.state_dict()[teacher_name]
    if teacher_param.shape != student_param.shape:
        print(f"Shape mismatch for {teacher_name}: teacher shape = {teacher_param.shape}, student shape = {student_param.shape}")
        continue
    student_param.copy_(teacher_param)



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def create_data_loader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4
    )
from datasets import Dataset,concatenate_datasets

# Load the  dataset



train_size = 0.8
train_dataset = dataset['train']#.sample(frac=train_size,random_state=200)
valid_dataset = dataset['validation']#.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset#.reset_index(drop=True)
test_dataset  = dataset['validation']#.reset_index(drop=True)
print(test_dataset["label"])

print("TRAIN Dataset: {}".format(train_dataset.shape))
print("VAL Dataset: {}".format(valid_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
validation_set = CustomDataset(valid_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

# n_labels = 2
batch_size = TRAIN_BATCH_SIZE

train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True)
## Create data loaders
# train_data_loader = create_data_loader(train_dataset, TRAIN_BATCH_SIZE)
# valid_data_loader = create_data_loader(valid_dataset, VALID_BATCH_SIZE)
# test_data_loader = create_data_loader(test_dataset, TEST_BATCH_SIZE)

# Define the optimizer and loss function
optimizer = AdamW(student_model.parameters(), lr=LEARNING_RATE)

def distillation_loss(y_student, y_teacher, labels, temperature):
    soft_targets = F.softmax(y_teacher / temperature, dim=1)
    log_probs = F.log_softmax(y_student / temperature, dim=1)
    loss = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)
    return loss



loss_vals = []
train_iterator = trange(num_epochs, desc="Epoch")
import torch
import torch.nn.functional as F
from transformers import glue_tasks_num_labels, glue_compute_metrics

def calculate_distillation_loss(teacher_logits, student_logits, targets, temperature):
    # Apply softmax to both teacher and student logits
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    
    # Compute the KL divergence loss
    loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')
    
    # Optionally, you can scale the loss by the temperature squared to match the original distillation loss formulation
    loss = loss * temperature * temperature
    
    return loss

from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
# Training loop
train_iterator = trange(num_epochs, desc="Epoch")
for epoch in train_iterator:
    print('############# Epoch {}: Training Start #############'.format(epoch))
    epoch_iterator = tqdm(train_loader, desc="Iteration")
    train_loss = 0
    val_loss = 0
    num_val_steps = 0
    y_true = []
    y_pred = []
    
    # Train the model
    student_model.train()
    for batch_idx, data in enumerate(epoch_iterator):
        optimizer.zero_grad()
        
        ids = data['ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        
        student_logits = student_model(ids, mask)
        
        teacher_logits = get_logits(teacher_model, ids, mask, token_type_ids)
        distillation_loss = calculate_distillation_loss(teacher_logits, student_logits, targets, TEMPERATURE)
        
        distillation_loss.backward()
        optimizer.step()
        
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (distillation_loss.item() - train_loss))
        
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(torch.argmax(student_logits, dim=1).cpu().numpy())
    
    # Evaluate the student model on the validation set
    student_model.eval()
    with torch.no_grad():
        for batch in val_loader:
            ids, mask, token_type_ids, labels = (
                batch['ids'].to(device),
                batch['mask'].to(device),
                batch['token_type_ids'].to(device),
                batch['targets'].to(device)
            )
            teacher_logits = get_logits(teacher_model, ids, mask, token_type_ids)
            student_logits = student_model(ids, mask)
            
            val_loss += calculate_distillation_loss(teacher_logits, student_logits, labels, TEMPERATURE).item()
            num_val_steps += 1
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.argmax(student_logits, dim=1).cpu().numpy())
    
    val_loss /= num_val_steps
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    glue_metric = glue_compute_metrics("rte", predicted_labels=y_pred, labels=y_true)
    
    wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "f1_score": f1, "mcc": mcc, "accuracy": acc, "glue_metric": glue_metric})
    
    print('Epoch {} - train_loss: {:.4f}, val_loss: {:.4f}'.format(epoch + 1, train_loss, val_loss))
    print('F1 Score: {:.4f}, Matthews Corrcoef: {:.4f}, Accuracy: {:.4f}'.format(f1, mcc, acc))
    print('Glue Metric Score:', glue_metric)
