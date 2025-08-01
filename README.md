# Project_Creating-Realistic-Images-with-GANs
## Realistic Image Generation using GANs on CIFAR-10


## Project Goal

To develop a Generative Adversarial Network (GAN) that generates realistic 32×32 color images by learning patterns from the CIFAR-10 dataset using adversarial training.

## Objective

1. Implement and train a GAN model to generate high-quality, realistic images resembling the CIFAR-10 dataset.

2. Monitor the performance through Discriminator and Generator losses.

3. Visualize generated samples at different training epochs to evaluate image realism and diversity.

## Requirements

1. Python 3.x

2. TensorFlow / PyTorch

3. NumPy

4. Matplotlib

5. Jupyter Notebook / Google Colab

## 📚 Dataset: CIFAR-10

--> CIFAR-10 is a labeled dataset consisting of 60,000 32×32 color images in 10 different classes:

1. Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

2. Training Set: 50,000 images

3. Test Set: 10,000 images

## Preprocessing Steps:

1. Normalized pixel values from [0, 255] to [-1, 1]

2. Reshaped to match GAN input dimensions

3. Shuffled and batched for training

## Model Architectures

1. Generator
   
--> Input: Random noise vector (e.g., 100-dim)

--> Architecture:

--> Dense layer → BatchNorm → ReLU

--> Reshape to 4×4×512

--> Series of Conv2DTranspose layers with BatchNorm and ReLU

--> Final Conv2DTranspose layer with tanh activation to generate a 32×32×3 image

2. Discriminator
   
--> Input: 32×32×3 image

--> Architecture:

--> Series of Conv2D layers with LeakyReLU and Dropout

--> Flatten → Dense → Sigmoid to classify real vs. fake

## Training Loop

Loss Function: Binary Cross Entropy


Optimizer: Adam (learning rate = 0.0002, beta1 = 0.5)


Epochs: 51


For each batch:


Train Discriminator on real and generated images


Train Generator to fool the Discriminator


## Training Stability Techniques

Batch Normalization in Generator


LeakyReLU activations in Discriminator


Dropout regularization


Using tanh and sigmoid at output layers


Optional: Label smoothing and noise injection
 
## Outputs


<img width="402" height="339" alt="image" src="https://github.com/user-attachments/assets/709c6ee3-4f7c-4f67-a422-5e273a356843" />


<img width="311" height="329" alt="image" src="https://github.com/user-attachments/assets/13e88f7f-ae93-489c-974a-755852cb0e51" />


<img width="338" height="338" alt="image" src="https://github.com/user-attachments/assets/b7a21e1f-14ae-44e3-88a4-4d45b8de42ed" />


<img width="310" height="336" alt="image" src="https://github.com/user-attachments/assets/40fa91dd-f20e-4c35-9245-5f562795d065" />


<img width="346" height="348" alt="image" src="https://github.com/user-attachments/assets/d0a42660-a460-4549-b7ee-615b85f9965a" />


## Observations & Learning

1. Early epochs generated noisy and indistinct images.

2. By epoch 30, some class-specific structure started to appear (e.g., shapes resembling airplanes, ships).

3. Final outputs show moderately realistic images, though further training or more advanced techniques (e.g., DCGAN, WGAN) could improve quality.

4. Generator and Discriminator losses fluctuated, indicating adversarial balance.

## Result

The GAN was able to generate visually realistic CIFAR-10-like images by epoch 50. It learned the distribution of real images and produced diverse outputs showing structured patterns.
