## PyTorch

Set up

Make your conda env have a kernel visible to jupyter notebook

```bash
conda activate ./envconda install ipykernelpython3 -m ipykernel install --user --name myenv --display-name "Conda (env)"
```

### Operations

`weights.view(dim1, dim2)` is equivalent to `weights.reshape(dim1, dim2)`

To numpy and back

```python
a = np.random.rand(4,3)b = torch.from_numpy(4)  # torch tensorb.numpy()  # numpy array
```

### Loading data

Creating a Python iterator through my images

```python
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)dataiter = iter(trainloader)
```

### Creating a model

Create a Linear layer (weights plus bias)

```python
from torch import nnnn.Linear(784, 256)
```

Computing softmax across columns

```python
nn.Softmax(dim=1)
```

Template for creating a neural net

```python
class Network(nn.Module):    def __init__(self):        super().__init__()        # Inputs to hidden layer linear transformation        self.hidden = nn.Linear(784, 256)        # Output layer, 10 units - one for each digit        self.output = nn.Linear(256, 10)        # Define sigmoid activation and softmax output         self.sigmoid = nn.Sigmoid()        self.softmax = nn.Softmax(dim=1)    def forward(self, x):        # Pass the input tensor through each of our operations        x = self.hidden(x)        x = self.sigmoid(x)        x = self.output(x)        x = self.softmax(x)        return x
```

Functional way (more commonly used as more flexible than sequential one)

```python
import torch.nn.functional as Fclass Network(nn.Module):    def __init__(self):        super().__init__()        # Inputs to hidden layer linear transformation        self.fc1 = nn.Linear(784, 256)        self.fc2 = ...        # Output layer, 10 units - one for each digit        self.output = nn.Linear(256, 10)    def forward(self, x):        # Hidden layer with sigmoid activation        x = F.sigmoid(self.fc1(x))        x = F.relu(self.fc2(x))        ...        # Output layer with softmax activation        x = F.softmax(self.output(x), dim=1)        return x
```

Sequential API (easy to use, less flexible)

```python
input_size = 784hidden_sizes = [128, 64]output_size = 10from collections import OrderedDictmodel = nn.Sequential(OrderedDict([                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),                      ('relu1', nn.ReLU()),                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),                      ('relu2', nn.ReLU()),                      ('output', nn.Linear(hidden_sizes[1], output_size)),                      ('softmax', nn.Softmax(dim=1))]))
```

### Losses in Pytorch

**Cross-entropy**

For classification, takes as input the logits, not the probabilities (obtained after softmax).

More convenient to use a log-softmax output instead (use `torch.exp(outputs)` to retrieve probabilities).

```python
nn.LogSoftmax(dim=1)  # compute log-softmax across columns nn.NLLLoss()  # negative log-likelihood loss
```

### Autograd

#### Backward pass

Module computing gradients. Autograd tracks operations happening on tensors, after registering them.

```python
x = torch.zeroes(1, requires_grad=True)
```

If you don't want gradient computations to happen:

```python
x = torch.zeroes(1, requires_grad=False)# ORwith torch.no_grad():  # useful when doing inference to speed up    y = x*2# OR disable globallytorch.set_grad_enabled(False)# If you want to change after creationx.requires_grad_(True)
```

Compute gradient of `z` with respect to graph leaves

```python
z.backward(tns)
```

Each tensor obtained from an operation has a `grad_fn` attribute telling what function generated the tensor. PyTorch knows how to compute gradients of each function, and is used during the backward pass.

When calling `.backward()` a backward pass is done and gradients are stored along the way.

#### Optimizing: changing the weights

```python
from torch import optim# Optimizers require the parameters to optimize and a learning rateoptimizer = optim.SGD(model.parameters(), lr=0.01)
```

Gradients are stored in memory when doing multiple backward passes. To avoid this:

```python
# Clear the gradients of all tensors, otherwise gradients are summed up at each pass !!# Clear at each pass, not only after declarationoptimizer.zero_grad()
```

Once you run one `.backward()` pass, you can update the weights, according to the optimizer you chose:

```python
optimizer.step()  # update parameters specified during optimizer creation
```

#### Remarks

- Useful to flatten or modify your tensor shape inside

- `.item()` of a scalar tensor returns a float value

- Change type of tensor `.type(torch.FloatTensor)`

- add a `.dropout(p=0.2)` layer on top of activation layer for regularization.

- Better to call `model(input)` than `model.forward(input)` (hooks are dispatched in _*call_* function, so they won't be called if you use `.forward()`)

- Debug: for validation make sure you are on evaluation mode `model.eval()`

- Debug: for training `model.train()`

### Inference and validation

#### Metrics

- `Tensor.topk` returns a tuple of top-k values, and top-k indices

- Accuracy
  
  ```python
  top_p, top_class = ps.topk(1, dim=1)equals = top_class == labels.view(*top_class.shape)# and not: equals = top_class == labelsaccuracy = torch.mean(equals.type(torch.FloatTensor))# torch.mean is not implemented for torch.ByteTensor, which is the initial type of the tensor equals
  ```

- Evalute your network with drop-out probability set to 0 with
  
  ```python
  with torch.no_grad():    # eval mode    model.eval()        # validation passes    # go back to train mode    model.train()
  ```
  
  ## Saving and Loading models
  
  Models contain a `state_dict` containing weights and biases matrices for the different layers.

- Basic Worflow:
  
  ```python
  # Save state dictionarytorch.save(model.state_dict(), 'checkpoint.pth')# reload the state dictionarystate_dict = torch.load('checkpoint.pth')# load state dictionary if it has similar architecturemodel.load_state_dict(state_dict)
  ```

- Save all necessary information to rebuild the trained model:
  
  ```python
  # Model infocheckpoint = {'input_size': 784,              'output_size': 10,              'hidden_layers': [each.out_features for each in model.hidden_layers],              'state_dict': model.state_dict()}torch.save(checkpoint, 'checkpoint.pth')# Example of reloading, with a custom class to build a networkdef load_checkpoint(filepath):    checkpoint = torch.load(filepath)    model = fc_model.Network(checkpoint['input_size'],                             checkpoint['output_size'],                             checkpoint['hidden_layers'])    model.load_state_dict(checkpoint['state_dict'])        return model    model = load_checkpoint('checkpoint.pth')
  ```

### ImageFolder

Import data

```python
dataset = datasets.ImageFolder('path/to/data', transform=transform)
```

with data in the following repo structure:

```textile
root/class1/image1_w_class1.png    /class1/image2_w_class2.png    ...    root/class2/image1_w_class2.png    /class2/image2_w_class2.png    ... 
```

Images are labelled with the directory name (here `class1` and `class2` ).

### Transforms

Process your images once loaded. Useful to standardize your images to be fed into a NN, but also to do data augmentation with random transformations.

[List of transforms available](%5Bhttps://pytorch.org/docs/0.3.0/torchvision/transforms.html%5D(https://pytorch.org/docs/0.3.0/torchvision/transforms.html)

```python
# Resizetransforms.Resize()# Croptransforms.CenterCrop()transforms.RandomResizedCrop()# ...# transforms.ToTensor()# Data augmentations techniquestransforms.RandomRotation(30)transforms.RandomResizedCrop(224)transforms.RandomHorizontalFlip()transforms.ToTensor()transforms.Normalize([0.5, 0.5, 0.5],                     [0.5, 0.5, 0.5])  # to normalize the 3 colors channels
```

Create a pipeline of these successive transforms:

```python
transform = transforms.Compose([transforms.Resize(255),                                transforms.CenterCrop(224),                                transforms.ToTensor()])
```

### Data Loaders

Once a dataset has been created an optionally transformed by a transform object, pass it through a `DataLoader` . It creates a generator with control over the size of batches, shuffling data after an epoch, etc.

```python
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)for images, labels in dataloader:    pass    images, labels = next(iter(dataloader))
```

### Transfer Learning

Use pre-trained model from `torchvision`:

```python
from torchvision import modelsmodel = models.densenet121(pretrained=True)
```

- pretrained on ImageNet (1 million samples, 1000 categories)

- Stack of convolutional layers (feature detectors) plus simple classifier on top

- Freeze parameters to avoid backprop
  
  ```python
  for param in model.parameters():    param.requires_grad = False
  ```

- Create your own `classifier` on top of the CNN stack, by replacing the huge initial `classifier` :
  
  ```python
  classifier = nn.Sequential(OrderedDict([                          ('fc1', nn.Linear(1024, 500)),                          ('relu', nn.ReLU()),                          ('fc2', nn.Linear(500, 2)),                          ('output', nn.LogSoftmax(dim=1))                          ]))    model.classifier = classifier
  ```

Standard pre-processing steps for models (like AlexNet, ResNet, etc.):

- 224x224 images

- Normalization: per channel, with means `[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]`

#### GPU training

- Define your loss criteria

- Define your optimizer only on the classifier parameters, so that backprop affects only this part

- Send model to CUDA (GPU memory) `model.to('gpu')`

- Trigger the computations
  
  - Generate images from the iterator `for inputs, labels in trainloader`
  
  - send images to GPU `inputs, labels = inputs.to('gpu'), labels.to('gpu')`
  
  - Usual steps, but run on GPU:
    
    - Run forward pass `.forward()`,
    
    - compute loss `loss=...`,
    
    - run backward pass `.backward()`,
    
    - update params `.step()`

```
Remarks:-   use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` instead of `'gpu'` to make your code hardware agnostic.-   Sometimes error with a `torch.cuda.FloatTensor` instead of `torch.FloatTensor` : make sure tensors used for the computations are either all on the GPU or all on the CPU.
```
