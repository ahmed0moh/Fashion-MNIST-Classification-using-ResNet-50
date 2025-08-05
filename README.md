# 👗 Fashion MNIST Classification with ResNet-50 (PyTorch + Jupyter Notebook)

This notebook fine-tunes a pre-trained *ResNet-50* model to classify images in the **Fashion MNIST** dataset. We unlock the first four layers for training, append two fully connected layers, and leverage parallel computing to accelerate training.

---

## 🔍 Project Overview

1. **Data Preparation**  
   - Load Fashion MNIST  
   - Resize to 16×16, normalize  
   - Create DataLoader for train & validation sets  

2. **Model Setup**  
   - Load resnet50(pretrained=True)  
   - Enable requires_grad=True for:  
     - conv1, bn1, relu, maxpool  
   - Freeze other layers  
   - Define custom FC head:  
     python
     class Model(nn.Module):
         def __init__(self, in_size, hidden_size, out_size):
             super().__init__()
             self.fc1 = nn.Linear(in_size, hidden_size)
             self.fc2 = nn.Linear(hidden_size, out_size)
         def forward(self, x):
             x = torch.relu(self.fc1(x))
             return self.fc2(x)
     
   - Replace model.fc = Model(2048, 512, n_classes)  

3. **Training Configuration**  
   - Device: cuda if available, else cpu  
   - Multi-thread & DataParallel:  
     python
     torch.set_num_threads(3)
     model = DataParallel(model).to(device)
     
   - Loss & Optimizer:  
     python
     criterion = nn.CrossEntropyLoss()
     optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8)
     if lr_scheduler:
         schedule = lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=5)
     
   - Train for n_epochs=25, batch_size=500  

4. **Utility Functions**  
   - ***plot_stuff(COST, ACC)***  
     Plots training loss (red, left y-axis) and accuracy (blue, right y-axis).
   - ***imshow_(inp, title)***  
     Converts a tensor to a NumPy image, denormalizes, and displays it.
   - ***result(model, x, y)***  
     Runs a single input through the model and prints mismatches.

5. **Training Loop**  
   - Records loss & accuracy per epoch  
   - Saves best model weights based on validation accuracy  
   - Prints learning rate, loss, and accuracy each epoch  

6. **Saving & Loading**  
   - Save checkpoint:
     ```python
     torch.save({
         'epoch': n_epochs,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'loss': loss_list,
         'accuracy': accuracy_list,
         'parameters': parameters
     }, 'FashionMNISTResNet50.pt')
     plot_stuff(loss_list, accuracy_list)
     ```
   - Load checkpoint:
     ```python
     checkpoint = torch.load('FashionMNISTResNet50.pt')
     model.load_state_dict(checkpoint['model_state_dict'])
     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
     loss_list, accuracy_list = checkpoint['loss'], checkpoint['accuracy']
     model.eval()
     ```

---

## 📊 Dataset & Preprocessing

Dataset: Fashion MNIST
28×28 grayscale images of 10 clothing categories (e.g., T-shirt, Trouser, Sneaker).

Transforms:

1. Resize images to 16×16 to reduce compute.


2. ToTensor: converts H×W×C to C×H×W in [0,1].


3. Normalize with mean=0.5, std=0.5 → pixel values ∈ [−1,1].



```python
IMAGE_SIZE = 16
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = FashionMNIST(..., train=True,  transform=transform)
val_set   = FashionMNIST(..., train=False, transform=transform)
```
DataLoaders:
```python
train_loader = DataLoader(train_set, batch_size=500, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_set,   batch_size=2500, shuffle=False)
```


---

## 🔧 Model Architecture

1. Base: torchvision.models.resnet50(pretrained=True)


2. Trainable Layers:
```python
for name, layer in model.named_children():
    layer.requires_grad = (name in ['conv1','bn1','relu','maxpool'])
```

3. Custom Head:
```python
class Head(nn.Module):
    def _init_(self, in_features, hidden_size, num_classes):
        super()._init_()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

n_classes = len(set(train_set.targets.numpy()))
model.fc = Head(2048, 512, n_classes)
```

4. Parallelism & Device:
```python
torch.set_num_threads(3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DataParallel(model).to(device)
```



---

## ⚙ Training Configuration

Hyperparameters:

epochs:        25
batch_size:    500
learning_rate: 1e-4
momentum:      0.8
LR Scheduler:  CyclicLR(base_lr=1e-3, max_lr=1e-2, step_size_up=5, mode="triangular2")

Criterion & Optimizer:
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
if use_scheduler:
    scheduler = lr_scheduler.CyclicLR(optimizer, **scheduler_params)
```


---

🔄 Utility Functions
```python
plot_stuff(COST, ACC)
```
Dual-axis plot: total loss (red, left y-axis) & accuracy (blue, right y-axis).
```python
imshow_(inp, title=None)
```
Convert a C×H×W tensor to H×W×C NumPy array, denormalize (ImageNet stats), and display with Matplotlib.
```python
result(model, x, y)
```
Predict a single sample (unsqueezed), compare with true y, and print mismatches.

```python
def plot_stuff(COST, ACC): ...
def imshow_(inp, title=None): ...
def result(model, x, y): ...
```
Detailed explanations are inline in the notebook.


---

## 📈 Training Loop & Checkpointing

1. Loop over n_epochs with tqdm progress bar.


2. Inner loop:

Forward pass → compute loss

Backward pass → optimizer.step(), optimizer.zero_grad()



3. Scheduler: scheduler.step() each epoch.


4. Validation: compute accuracy on val_loader.


5. Track & Save best model weights (highest val accuracy).


```python
checkpoint = {
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss_list,
    'accuracy': accuracy_list,
    'params': training_params
}
torch.save(checkpoint, 'FashionMNISTResNet50.pt')

     loss_list, accuracy_list = checkpoint['loss'], checkpoint['accuracy']
     model.eval()
```

