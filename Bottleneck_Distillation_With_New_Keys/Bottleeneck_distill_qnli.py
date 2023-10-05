import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from evaluate import load
import wandb
wandb.login(key="a6d93dd680e09e7dddae91cf4cf664991f59025f")

wandb.init(
    # set the wandb project where this run will be logged
    project="Bottleneck Distillation Keys not Carried",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 3e-5,
    "architecture": "Bottleneck",
    "dataset": "QNLI",
    "epochs": 100,
    }
)
MAX_LEN = 512
n_labels = 3
num_epochs = 5
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
LEARNING_RATE = 3e-5
LEARNING_RATE2= 0.1
TEMPERATURE = 1.0
batch_size= TRAIN_BATCH_SIZE
glue_metric = load('glue', 'qnli')
dataset = load_dataset("glue", "qnli")
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text1 = dataframe['question']#dataframe.sentence
        self.text2 = dataframe['sentence']
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

def empty_init(*shape):
    #return torch.empty(shape)
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def collect_embeddings(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b t -> h b t d', d = dim) # extend(repeat) indicies with head dimensin
    embeds = repeat(embeds, 'h c d -> h b c d', b = batch) # extend(repeat) key embeddings with batch dimension
    return embeds.gather(2, indices) # gather closest keys from codebook based on indeces

class VectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        n_heads,
        heads_dim,
        codebook_size,
        decay
    ):
        super().__init__()
        self.input_dim=input_dim
        self.n_heads=n_heads
        self.decay = decay
        self.codebook_size = codebook_size

        key_embed = empty_init(n_heads,codebook_size,heads_dim) # init codebooks
        #print("key embed initted ",key_embed[0].shape)

        self.register_buffer('key_embed', key_embed)  # register as non weight, but still part of model
        self.register_buffer('key_embed_avg', key_embed.clone()) # register as non weight, but still part of model

    def forward(
        self,
        x,
        key_optim
    ):

        x = x.float()
        shape, dtype = x.shape, x.dtype

        if self.n_heads>1:
            #x = rearrange(x, 'b t d -> b h t d', h = self.n_heads) Segment on tokens???
            ein_rhs_eq = 'h b t d' # h-head, b-batch, t-token, d-heads dimension
            x = rearrange(x, f'b t (h d) -> {ein_rhs_eq}', h = self.n_heads) # segment input into heads


        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, 'h ... d -> h (...) d') # merge the batch and token dimensions

        emb = self.key_embed
        #print("input shape ",shape, "flatten shape ", flatten.shape, "embed shape ", emb.shape)

        dist = -torch.cdist(flatten, emb, p = 2)  # calculate euclidean distance

        #print("dist ",dist.shape)

        emb_ind = dist.argmax(dim= -1) # save indices of closest keys for each head
        emb_onehot = F.one_hot(emb_ind, self.codebook_size).type(dtype) # one hot encoding for ( head, token, codebook index 1hot )
        #print("emb ind ", emb_ind.shape)
        emb_ind = emb_ind.view(*shape[:-1])

        #print("emb_onehot ", emb_onehot.shape)

        quantized = collect_embeddings(emb_ind, emb) # collect closest key for each head

        if key_optim:
            emb_sum = einsum('h n d, h n c -> h c d', flatten, emb_onehot) # elementwise multiplication and summation over axis n
            self.key_embed.data.lerp_(emb_sum, self.decay)

        #print("ke ",self.key_embed.data[0][0])

        quantized = rearrange(quantized, 'h b t d -> b t (h d)' , h=self.n_heads) # concatenate the segments back together
        emb_ind = rearrange(emb_ind, 'h b n -> b n h', h=self.n_heads) # reshape indice tensor

        return quantized, emb_ind

import torch

from torch import nn, einsum
from einops import rearrange, repeat

# from vq_ema import VectorQuantize

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d
    
class DiscreteKeyValueBottleneck(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_key_segments = 64, # number of key segments
        codebook_size = 4096,   # number of different discrete keys in bottleneck codebook
        dim_key = 12,        # dimension of the key segments
        dim_value = 12,
        decay = 1,
        encoder = None,
        decoder = None,
        pool_before = False,
        pooling_type = "cls",
        n_labels = n_labels,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pool_before = pool_before
        self.pooling_type = pooling_type
        self.n_labels = n_labels

        assert (dim % num_key_segments) == 0, 'embedding dimension must be divisible by number of codes'
        assert decoder =='mlp' or dim_value==self.n_labels, 'if decoder is values_softmax dim_values must equal to number of labels'
        assert decoder =='values_softmax' or (num_key_segments*dim_value)==768, 'if decoder is mlp num_key_segments*dim_value must equal to encoder output dim'

        self.vq = VectorQuantize(
            input_dim = dim,
            n_heads = num_key_segments,
            heads_dim = dim_key,
            codebook_size = codebook_size,
            decay = decay
        )

        self.values = nn.Parameter(torch.randn(num_key_segments, codebook_size, dim_value))

    def forward(
        self,
        x,
        mask,
        #token_type_ids,
        key_optim,
        **kwargs
    ):

        if exists(self.encoder):
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x, mask,**kwargs)
                if self.pool_before:
                    if self.pooling_type =="cls":
                        x = x[1]
                    if self.pooling_type =="mean":
                        x = x[0].mean(dim=1)
                    x = rearrange(x, 'b h -> b 1 h')
                    #print(" x: ", x.shape, "\n values ", x)
                else:
                    x = x[0]

        vq_out = self.vq(x, key_optim)

        if key_optim: # if we are optimizing keys with ema, break forward here
            return None

        quantized, memory_indices = vq_out

        #print("quantized shape ",quantized.shape, " /n memory_indices shape :", memory_indices.shape)
        #print(" values shape ", self.values.shape)

        if memory_indices.ndim == 2:
            memory_indices = rearrange(memory_indices, '... -> ... 1')

        memory_indices = rearrange(memory_indices, 'b n h -> b h n')

        values = repeat(self.values, 'h n d -> b h n d', b = memory_indices.shape[0])
        #print("values after reshape ", values.shape)

        memory_indices = repeat(memory_indices, 'b h n -> b h n d', d = values.shape[-1])
        #print("memory ind ",memory_indices.shape)

        memories = values.gather(2, memory_indices)
        #print("memories ",memories.shape)

        memories = rearrange(memories, 'b h n d -> b n h d')
        #print("memories ",memories.shape)

        if self.decoder =='mlp':
            memories = rearrange(memories, 'b n h d -> b n (h d)')
        #print("memories ", memories.shape)

        return memories#flattened_memories

from transformers import BertModel
from torch import nn
from einops import rearrange

# from dkv_bn import DiscreteKeyValueBottleneck

class BERTwithBottleNeck(nn.Module):
    def __init__(self,pool_before,pooling_type,n_labels):
        super(BERTwithBottleNeck, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.pool_before = pool_before
        self.pooling = pooling_type
        self.n_labels = n_labels

        self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
            encoder = self.encoder,   # pass the frozen encoder into the bottleneck
            decoder = 'mlp', # type of decoder: 1 layer mlp or values_softmax(non parametric)
            dim = 768,                # input dimension
            num_key_segments = 64, # number of key segments
            codebook_size = 4096,   # number of different discrete keys in bottleneck codebook
            dim_key = 12,        # dimension of the key segments
            dim_value = 12,        # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
            decay = 0.8,              # the exponential moving average decay, lower means the keys will change faster
            pool_before = self.pool_before, # boolean flag whether to pool before or after bottleneck
            pooling_type = self.pooling, # type of pooling : cls or mean
            n_labels = self.n_labels
        )

        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)

    def forward(self, ids, mask,token_type_ids, key_optim=False):
        outputs = self.enc_with_bottleneck(ids, mask=mask,token_type_ids=token_type_ids,key_optim=key_optim)
        if key_optim:
           return None # Finish forward pass here during key optimization

        #
        if not self.pool_before:
            if self.pooling == "cls":
               outputs = outputs[:,0] # Pool by CLS token here
            if self.pooling == "mean":
               outputs = outputs.mean(dim=1) # Pool by mean of token dim here
        if self.pool_before:
           outputs = rearrange(outputs, 'b 1 d -> b d')

        if self.enc_with_bottleneck.decoder=='mlp':
            dropout_output = self.l2(outputs)
            logits = self.l3(dropout_output)
        if self.enc_with_bottleneck.decoder=='values_softmax':
            logits = outputs.mean(dim=1)

        return logits

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

class DistilBERTwithBottleNeck(nn.Module):
    def __init__(self, pool_before, pooling_type, n_labels):
        super(DistilBERTwithBottleNeck, self).__init__()
        self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pool_before = pool_before
        self.pooling = pooling_type
        self.n_labels = n_labels

        self.enc_with_bottleneck = DiscreteKeyValueBottleneck(
            encoder=self.encoder,   # pass the frozen encoder into the bottleneck
            decoder='mlp', # type of decoder: 1 layer mlp or values_softmax(non parametric)
            dim=768,                # input dimension
            num_key_segments=64, # number of key segments
            codebook_size=4096,   # number of different discrete keys in bottleneck codebook
            dim_key=12,        # dimension of the key segments
            dim_value=12,        # dimension of the value segments, should equal to n_labels if values softmax or dim_key if mlp
            decay=0.8,              # the exponential moving average decay, lower means the keys will change faster
            pool_before=self.pool_before, # boolean flag whether to pool before or after bottleneck
            pooling_type=self.pooling, # type of pooling : cls or mean
            n_labels=self.n_labels
        )

        self.l2 = nn.Dropout(0.1)
        self.l3 = nn.Linear(768, self.n_labels)

    def forward(self, ids, mask, key_optim=False):
        outputs = self.enc_with_bottleneck(ids, mask=mask, key_optim=key_optim)
        if key_optim:
            return None  # Finish forward pass here during key optimization

        if not self.pool_before:
            if self.pooling == "cls":
                outputs = outputs[:, 0]  # Pool by CLS token here
            if self.pooling == "mean":
                outputs = outputs.mean(dim=1)  # Pool by mean of token dim here
        if self.pool_before:
            outputs = rearrange(outputs, 'b 1 d -> b d')

        if self.enc_with_bottleneck.decoder == 'mlp':
            dropout_output = self.l2(outputs)
            logits = self.l3(dropout_output)
        if self.enc_with_bottleneck.decoder == 'values_softmax':
            logits = outputs.mean(dim=1)

        return logits
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
teacher_model = BERTwithBottleNeck(n_labels = n_labels,pooling_type='mean',pool_before=False).to(device)
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
optimizer = AdamW([
    {'params': student_model.enc_with_bottleneck.parameters(), 'lr': LEARNING_RATE},
    {'params': student_model.l3.parameters(), 'lr': LEARNING_RATE2}
], lr=LEARNING_RATE)
def distillation_loss(y_student, y_teacher, labels, temperature):
    soft_targets = F.softmax(y_teacher / temperature, dim=1)
    log_probs = F.log_softmax(y_student / temperature, dim=1)
    loss = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)
    return loss



student_model.train()
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

train_iterator = trange(num_epochs, desc="Epoch")
for epoch in train_iterator:
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    epoch_iterator = tqdm(train_loader, desc="Iteration")
    train_loss = 0
    val_loss = 0
    num_val_steps = 0
    y_true = []
    y_pred = []
    
    # Train the model
    for batch_idx, data in enumerate(epoch_iterator):
        optimizer.zero_grad()
        
        ids = data['ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        
        student_logits = student_model(ids, mask, key_optim=False)
        
        # Calculate distillation loss
        teacher_logits = get_logits(teacher_model, ids, mask, token_type_ids)
        distillation_loss = calculate_distillation_loss(teacher_logits, student_logits, targets, TEMPERATURE)
        
        distillation_loss.backward()
        optimizer.step()
        
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (distillation_loss.item() - train_loss))
        
        # Collect true and predicted labels for evaluation
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(torch.argmax(student_logits, dim=1).cpu().numpy())
        wandb.log({"train_loss": train_loss, "epoch": epoch+1})

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
            student_logits = student_model(ids, mask, key_optim=False)
            
            val_loss += calculate_distillation_loss(teacher_logits, student_logits, labels, TEMPERATURE).item()
            num_val_steps += 1
            
            # Collect true and predicted labels for evaluation
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.argmax(student_logits, dim=1).cpu().numpy())


    # glue_metric = glue_compute_metrics("qnli", predicted_labels=y_pred, labels=y_true)
    val_loss /= num_val_steps
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    wandb.log({"f1_score": f1, "epoch": epoch+1})
    wandb.log({"acc_score": acc, "epoch": epoch+1})
    print('Epoch {} - train_loss: {:.4f}, val_loss: {:.4f}'.format(epoch + 1, train_loss, val_loss))
    print('F1 Score: {:.4f}, Matthews Corrcoef: {:.4f}, Accuracy: {:.4f}'.format(f1, mcc, acc))
    # print('Glue Metric Score:',glue_metric)