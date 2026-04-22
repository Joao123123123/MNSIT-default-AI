# MNSIT-default-AI
I made this AI in december 2024
MNIST Neural Network from Scratch
This project is an artificial neural network implementation built entirely from scratch using only Python and NumPy. The goal is to recognize handwritten digits (0-9) using the classic MNIST dataset.
Unlike modern implementations that rely on high-level frameworks like TensorFlow or PyTorch, this network was developed "manually." This includes the core mathematics of Backpropagation, manual gradient calculations, and custom weight management via .npy files.

The network architecture is:
Input layer: 784 neurons(representing the pixels in the 28x28 images)
Second layer: 128 neurons
Third layer: 64 neurons
Output layer: 10 neurons

"Techniques"
I used softmax and relu. I changed the common relu derivative to 0.1 when x<0 so the neurons dont die if my weight goes too negative.  I used gradient clipping to prevent my weight values from exploding and Manual learning rate decay. PIL was used to process the images.

After tuning and debugging, the network achieved:
Peak Accuracy:~99%
Consistent accuracy: The network sits at a stable ~94% to ~96% accuracy

File Structure:
weights.npy, weights_2.npy, weights_3.npy: Trained weight matrices for each layer.
bias1.npy, bias2.npy, bias3.npy: Adjusted biases learned by the model.
main.py: The core script containing the training loop, forward pass, and backpropagation logic.

How to Run
Ensure the MNIST dataset is organized into folders labeled 0 through 9.
Update the dataset path in the script to point to your local files.
DOWNLOAD THE PIL AND NUMPY LIBRARIES
Run the script. The system will automatically load existing weights or generate new ones if the .npy files are missing.

OBS:
The greatest challenge of this project was finding the perfect amount of neurons for my layers. I spent a long time training an ai with 16 neurons and 22 neurons on the second and third layers, respectively, and was wondering why its accuracy wasnt increasing. This 'panic' made me add many extra mesures such as: Changing from sigmoid to relu(I thought that maybe the sigmoid was squishing my derivative due to its graph flattening out when near 1), Changing from mean squared error to the cross entropy loss(Added competition to the output neurons value as I thought thats what was keeping my ai from learning) and adding that 0.1 to the relu derivative to prevent neurons from dying. Finally, when I decided to read further about AI's I decided to try and change my AIs neuron architecture, the moment I changed the second layer's neuron amount to 64 the AI started working. From then on I found, empirically, that the optimal amount of neurons for my neural network was 784 : 128 : 64 : 10.
