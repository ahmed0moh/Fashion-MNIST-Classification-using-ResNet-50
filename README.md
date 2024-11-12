# Fashion-MNIST-Classification-using-ResNet-50
This project focuses on classifying images from the Fashion MNIST dataset using a pre-trained ResNet-50 model. The first four layers of the ResNet-50 model are made trainable to fine-tune the network for better performance on the Fashion MNIST dataset. Additionally, two fully connected layers are added to the model. Parallel computing is enabled to speed up the training process.

## Steps Involved

1. **Data Preparation**
   - Load the Fashion MNIST dataset.
   - Preprocess the images (normalization, resizing, etc.).

2. **Model Setup**
   - Load the pre-trained ResNet-50 model.
   - Modify the model to make the first four layers trainable.
   - Add two fully connected layers:
     - The first layer takes 2048 inputs and outputs 512.
     - The second layer takes 512 inputs and outputs the number of classes.

3. **Training**
   - Enable parallel computing to utilize multiple GPUs/CPUs.
   - Compile the model with appropriate loss function and optimizer.
   - Train the model on the Fashion MNIST dataset.

4. **Saving Model**
   - Saving Model Parameters and State in PyTorch.
