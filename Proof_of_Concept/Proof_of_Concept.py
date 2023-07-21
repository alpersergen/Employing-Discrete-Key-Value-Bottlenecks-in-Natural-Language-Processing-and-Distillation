import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import torch
from torchvision.datasets import MNIST
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch import nn, einsum
from einops import rearrange, repeat
from tqdm import tqdm, trange

# from vq_ema import VectorQuantize

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# main class

class DiscreteKeyValueBottleneck(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_memories,
        num_memory_codebooks,
        encoder = None,
        dim_memory = None,
        pool_before = False,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.pool_before = pool_before
        assert (dim % num_memory_codebooks) == 0, 'embedding dimension must be divisible by number of codes'
        self.vq = VectorQuantize(
            input_dim = dim,
            n_heads = num_memory_codebooks,
            heads_dim = dim_memory,
            codebook_size = num_memories,
            decay = 0.8
        )

        dim_memory = default(dim_memory, dim // num_memory_codebooks)
        self.values = nn.Parameter(torch.randn(num_memory_codebooks, num_memories, dim_memory))

    def forward(
        self,
        x,
        key_optim,
        **kwargs
    ):
        if exists(self.encoder):
             self.encoder.eval()
             with torch.no_grad():
                 x = self.encoder(x)
#        x = self.encoder(x)
        x = F.relu(x)

        if self.pool_before:

          x = x[1]
          x = rearrange(x, 'b h -> b 1 h')
          #print(" x: ", x.shape, "\n values ", x) 
                    
        vq_out = self.vq(x, key_optim)
        
        if key_optim: # if we are optimizing keys with ema, break forward here
            return None #torch.empty_like(x)
            
        quantized, memory_indices = vq_out
        
        # print("quantized shape ",quantized.shape, " /n memory_indices shape :", memory_indices.shape)
        # print(" values shape ", self.values.shape)
        
        if memory_indices.ndim == 2:
            memory_indices = rearrange(memory_indices, '... -> ... 1')

        #memory_indices = rearrange(memory_indices, 'b n h -> b h n')

        values = repeat(self.values, 'h n d -> b h n d', b = memory_indices.shape[0])
        # print("values after reshape ", values.shape)
        
        memory_indices = repeat(memory_indices, 'b h n -> b h n d', d = values.shape[-1])
        # print("memory ind ",memory_indices.shape)

        memories = values.gather(2, memory_indices)
        # print("memories ",memories.shape)

        flattened_memories = rearrange(memories, 'b h n d -> b n (h d)')
        #print("flattened memories ", flattened_memories.shape)
        # print('flattenmem',flattened_memories.shape)
        return flattened_memories
def empty_init(*shape):
    #return torch.empty(shape)
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t
    
def collect_embeddings(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    # print(batch,dim)
    # print(indices.shape)
    indices=indices.unsqueeze(-1)
    indices = repeat(indices, 'h b p -> h b p d', d = dim) # extend(repeat) indicies with head dimensin
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
        #print("key embed initted ",key_embed[0])
        
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
            # print(x.shape)  # (b (hw))
            ein_rhs_eq = 'h b d' # h-head, b-batch, p-pıxel, d-heads dimension
            x = rearrange(x, f'b (h d) -> {ein_rhs_eq}', h = self.n_heads) # segment input into heads
        
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, 'h ... d -> h (...) d') # merge the batch and token dimensions         
        
        emb = self.key_embed
        # print("input shape ",shape, "flatten shape ", flatten.shape, "embed shape ", emb.shape)

        dist = -torch.cdist(flatten, emb, p = 2)  # calculate euclidean distance
        
        #print("dist ",dist.shape)        
        
        emb_ind = dist.argmax(dim= -1) # save indices of closest keys for each head
        emb_onehot = F.one_hot(emb_ind, self.codebook_size).type(dtype) # one hot encoding for ( head, token, codebook index 1hot )
        # print("emb ind ", emb_ind.shape)
        emb_ind = emb_ind.view(*shape[:-1])
        
        # print("emb_onehot ", emb_onehot.shape)
        
        quantized = collect_embeddings(emb_ind, emb) # collect closest key for each head
        
        if key_optim:
            emb_sum = einsum('h n d, h n c -> h c d', flatten, emb_onehot) # weigthed sum of embeddings
            self.key_embed.data.lerp_(emb_sum, self.decay)
        
        #print("ke ",self.key_embed.data[0][0])
        
        quantized = rearrange(quantized, 'h b p d -> b p (h d)' , h=self.n_heads) # concatenate the segments back together
        emb_ind = rearrange(emb_ind, 'h b -> b h', h=self.n_heads) # reshape indice tensor
        
        return quantized, emb_ind
        
        
        

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes, num_memory_codebooks, num_memories, n_heads, heads_dim, emb_dim, decay):

        super(MLP, self).__init__()
        self.encoder = nn.Linear(input_dim, emb_dim)
        self.bottleneck = DiscreteKeyValueBottleneck(
            encoder = self.encoder,
            dim=hidden_dim,
            num_memory_codebooks=64,
            num_memories=8,
            dim_memory= 12,
            decay = 0.1,
            pool_before= False
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)
        

    def forward(self, x, key_optim=False):
        x =  rearrange(x, 'b c h w -> b (h w) c')
        # print(x.shape)
        x = x.squeeze(-1)
        output = self.bottleneck(x, key_optim=key_optim)
        # print('output',output.shape)
        if key_optim:
           return None # Finish forward pass here during key optimization
        #emb_ind = self.bottleneck(x, key_optim=key_optim)
        output = self.decoder(output)
        output = output.squeeze(1)
        # print('mlp output',output.shape)
        return output



def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch in train_loader:
        data, target = batch
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def discrete_key_init(training_loader,model,device):
    # train_iterator = trange(n_epochs, desc="Epoch")
    # for epoch in train_iterator:
    # optim_iterator = tqdm(training_loader, desc="Iteration")
    for batch in training_loader:
        data, target = batch
        data, target = data.to(device), target.to(device)
        model(data, key_optim=True)        
    return model

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            data, target = batch
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    return loss, accuracy


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch
            data, target = data.to(device), target.to(device)

            output = model(data)
            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 28 * 28  # MNIST image size
hidden_dim = 768
output_dim = 10  # Number of classes in MNIST-10 dataset
num_classes = 32
num_memory_codebooks = 8
num_memories = num_classes * num_memory_codebooks
n_heads = 4
heads_dim = hidden_dim // n_heads
decay = 0.99
emb_dim = 768
# key_optim = False
model = MLP(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_classes=num_classes,
    num_memory_codebooks=num_memory_codebooks,
    num_memories=num_memories,
    n_heads=n_heads,
    heads_dim=heads_dim,
    emb_dim=emb_dim,
    decay=decay,
    # key_optim=key_optim,
  ).to(device)
import torch
from torchvision import datasets, transforms


# Load the MNIST dataset
train_dataset = datasets.MNIST('mnist_data/', train=True, download=True, transform=ToTensor())

# Create a dictionary to store the separated training examples
class_data_dict = {}

# Iterate over the training dataset
for data, label in train_dataset:
    # Create a tuple of the input data and target label
    example = (data, label)
    
    # Append the example to the corresponding class in the dictionary
    if label not in class_data_dict:
        class_data_dict[label] = [example]
    else:
        class_data_dict[label].append(example)
print(len(class_data_dict))
#print(class_data_dict[9][:10])
Dataset_9 = class_data_dict[9]
Dataset_8 = class_data_dict[8]
Dataset_7 = class_data_dict[7]
Dataset_6 = class_data_dict[6]
Dataset_5 = class_data_dict[5]
Dataset_4 = class_data_dict[4]
Dataset_3 = class_data_dict[3]
Dataset_2 = class_data_dict[2]
Dataset_1 = class_data_dict[1]
Dataset_0 = class_data_dict[0]




train_dataset = train_dataset #MNIST(root='./', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root='mnist_data/', train=False, transform=ToTensor())

# Split train dataset into train and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


best_val_accuracy = 0.0
key_epochs = 15

for epoch in range(key_epochs):
    print(f"Epoch {epoch + 1}/{key_epochs}")
    discrete_key_init(training_loader= train_loader, model=model, device=device)
epoch_num = 100
for i in range(10) :
  batch_size = 16
  # train_dataset = DataLoader(class_data_dict[i], batch_size=batch_size, shuffle=False)
  
  train_dataset = class_data_dict[i] #MNIST(root='./', train=True, transform=ToTensor(), download=True)
  test_dataset = MNIST(root='mnist_data/', train=False, transform=ToTensor())

  # Split train dataset into train and validation sets
  train_size = int(0.8 * len(train_dataset))
  val_size = len(train_dataset) - train_size
  train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

  batch_size = 16
  # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size)
  test_loader = DataLoader(test_dataset, batch_size=batch_size)


  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  for epoch in range(epoch_num):
    print(f"Epoch {epoch + 1}/{epoch_num}")
    train(model, train_loader, criterion, optimizer, device)
    
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pt")
        print("Best model saved!")
    
test_accuracy = test(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")
for i in range(10) :
  batch_size = 16
  test_loader = DataLoader(class_data_dict[i], batch_size=batch_size, shuffle=False)

  # Evaluate the model on the test dataset
  model.eval()
  total_correct = 0
  total_samples = 0

  with torch.no_grad():
      for batch in test_loader:
          images, labels = batch
          images = images.to(device)
          labels = labels.to(device)

          outputs = model(images)
          _, predicted = torch.max(outputs, dim=1)
          total_samples += labels.size(0)
          total_correct += (predicted == labels).sum().item()

  accuracy = total_correct / total_samples * 100
  print(f"Test Accuracy: {accuracy:.2f}%",'Dataset is',i)
test_dataset = MNIST(root='mnist_data/', train=False, transform=ToTensor())

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_accuracy = test(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")
