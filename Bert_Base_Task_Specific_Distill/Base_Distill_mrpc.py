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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, BertModel, AdamW
from sklearn.metrics import f1_score
from tqdm import tqdm

from datasets import load_dataset


# Initialize Weights and Biases

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import wandb
# Initialize Weights and Biases
wandb.login(key="a6d93dd680e09e7dddae91cf4cf664991f59025f")

wandb.init(
    # set the wandb project where this run will be logged
    project="Only base knowledge-distillation",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 3e-5,
    "architecture": "Bottleneck with base networks only distilation",
    "dataset": "mrpc",
    "epochs": 100,
    }
)

# Load RTE dataset
dataset = load_dataset("glue", "mrpc")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
val_data = dataset["validation"]
# Initialize tokenizer and constants
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MAX_LEN = 512
n_labels = 2
num_epochs = 5
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
LEARNING_RATE = 3e-5
LEARNING_RATE2= 0.1
TEMPERATURE = 1.0

num_classes =  n_labels # Binary classification (entailment vs. non-entailment)
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
train_data = CustomDataset(train_dataset, tokenizer, MAX_LEN)
val_data = CustomDataset(val_dataset, tokenizer, MAX_LEN)

# DataLoaders
train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=VALID_BATCH_SIZE, shuffle=False)

# Define teacher and student models
teacher_model = BertModel.from_pretrained('bert-base-uncased').cuda()
student_model = DistilBertModel.from_pretrained('distilbert-base-uncased').cuda()


def get_logits(model, ids, mask, token_type_ids):
    outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
    
    logits = outputs
    
    return logits

#def get_logits(model, ids, mask, token_type_ids):
#    outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
#    
#    pooled_output = outputs.pooler_output
#    logits = model.cls(pooled_output)  # cls is the classifier head for BERT models
#    
#    return logits

# Optimizer and loss function





# Load pretrained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').cuda()

# Define a simple classifier on top of BERT
class Classifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(Classifier, self).__init__()
        self.bert_model = bert_model
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(bert_output[:, 0, :])  # Using the [CLS] token representation for classification
        return logits

# Initialize the classifier and move it to the GPU
classifier = Classifier(bert_model, num_classes).cuda()

# Define optimizer and loss function
optimizer = AdamW(classifier.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
# class CustomDataset(Dataset):
#     def __init__(self, data, tokenizer, max_len):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         item = self.data[index]
#         text1 = item['sentence1']
#         text2 = item['sentence2']
#         label = item['label']

#         inputs = self.tokenizer.encode_plus(
#             text1,
#             text2,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             truncation=True,
#             padding='max_length',
#             return_attention_mask=True,
#             return_tensors='pt'
#         )

#         input_ids = inputs['input_ids'].squeeze()
#         attention_mask = inputs['attention_mask'].squeeze()

#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'targets': label
#         }

# # Usage example
# train_data = CustomDataset(train_dataset, tokenizer, MAX_LEN)
# val_data = CustomDataset(val_dataset, tokenizer, MAX_LEN)

# # ... Rest of the code remains the same ...

# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
teacher_model = classifier
# Training loop
teacher_model.train()

for epoch in range(5):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/5"):
        input_ids, attention_mask, labels = batch["input_ids"].cuda(), batch["attention_mask"].cuda(), batch["targets"].cuda()
        
        optimizer.zero_grad()
        logits = classifier(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += len(labels)

    average_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    print(f"Epoch [{epoch+1}/5], Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

optimizer = AdamW(student_model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
for teacher_name, teacher_param in teacher_model.named_parameters():
    if teacher_name not in student_model.state_dict():
        print(f"Ignoring {teacher_name} as it is not present in the student model")
        continue
    student_param = student_model.state_dict()[teacher_name]
    if teacher_param.shape != student_param.shape:
        print(f"Shape mismatch for {teacher_name}: teacher shape = {teacher_param.shape}, student shape = {student_param.shape}")
        continue
    student_param.copy_(teacher_param)

for epoch in range(NUM_EPOCHS):
    student_model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['targets']
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        targets = targets.cuda()
        
        optimizer.zero_grad()
        logits = student_model(input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)  # Use mean pooling
        
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_loader)
    
    # Log training loss to wandb
    wandb.log({"train_loss": average_loss, "epoch": epoch+1})
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {average_loss:.4f}")
    
    # Validation
    student_model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['targets']
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            
            logits = student_model(input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)  # Use mean pooling
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    f1 = f1_score(all_targets, all_preds)  # Calculate F1 score
    acc = accuracy_score(all_targets, all_preds)
    # Log F1 score to wandb
    wandb.log({"f1_score": f1, "epoch": epoch+1})
    wandb.log({"acc_score": acc, "epoch": epoch+1})
    print(f"F1 Score: {f1:.4f}")
    print(f"acc Score: {acc:.4f}")