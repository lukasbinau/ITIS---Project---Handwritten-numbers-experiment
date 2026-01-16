## Handwritten Digit Recognition with Convolutional Neural Networks



**Overview**



This project investigates handwritten digit recognition using convolutional neural networks trained on the MNIST dataset.  

The primary focus is to study how model depth(number of hidden fully connected linear layers) affects model accuracy, particularly when models are evaluated on different kinds of degraded handwritten numbers. We trained 10 neural network models with increasing depth under identical conditions to ensure a fair comparison.







**Project Objectives**



\- Train multiple CNN architectures with varying numbers of hidden layers

\- Compare model performance under controlled training conditions

\- Evaluate robustness when testing on degraded handwritten digit images







**Neural Network Architecture**



All models share the same convolutional layers and differ only in the number of fully connected hidden linear layers.



Convolutional layers:

\- Input: 1 × 28 × 28 grayscale image

\- Convolution: kernel size 5 × 5

\- Activation: ReLU

\- Pooling: Max pooling

\- Output size: 320



Fully connected layers:

\- Variable number of hidden layers (1–10)

\- Hidden dimension: 64

\- Dropout: 0.25(25%)

\- Output layer: 10 classes (digits 0–9)



Each model "ModelN" corresponds to "N" hidden fully connected layers.







**Training Setup**



All models were trained using identical hyperparameters:



\- Dataset: MNIST handwritten numbers (60.000)

\- Loss function: CrossEntropyLoss

\- Optimizer: Adam

\- Learning rate: 0.001

\- Batch size: 128

\- Epochs: 10

\- Device: GPU(RTX 4070)



To ensure reproducibility, all experiments use:

\- Fixed random seeds for initializing weights, dropout and shuffle(batches)

\- Deterministic CUDA settings 







**Preprocessing function**



The preprocessing function is designed to balance MNIST compatibility while preserving potential degradation of the test data:



\- Convert image to grayscale

\- Invert image if background is bright

\- If image is already 28×28 no size formatting

\- For larger images:

&nbsp; - Gentle contrast 

&nbsp; - Soft box detection

&nbsp; - Padding to square

&nbsp; - Resizing to 28×28







**Testing**



While training:

\- Models are evaluated on the MNIST test set

\- Test loss and accuracy are tracked per epoch

\- Loss curves are saved and plotted for comparison



Testing the models:

\- Models can be tested on custom handwritten digit datasets

\- Datasets may contain degraded digits

\- If filenames has the correct filename format(8\_example.png) accuracy is computed automatically







**Project folder structure**



-main.py: Model code for training and evaluating

-app.py: Simple UI using the StreamLit framework, for easy testing of custom datasets

-run\_app.bat: Windows launcher for the UI

-data/: MNIST dataset

-Different\_test\_data/: Custom test datasets

-model\_hidden\_layers\_(1-10).pth: Saved model weights

-loss\_history.csv: Training/testing loss history

-test\_loss\_all\_models.png: Loss comparison plot

-MNIST\_Model\_Usage\_Guide: List of commands to run main.py file from a terminal

-README.md







**How to use the program**



There are two main ways to use the program:

-Using a terminal

-Using the UI

Everything can be done from the terminal, but the UI is only for evaluating custom test data and visualization.



From the terminal(all terminal commands should be executed from the project folder):



1. Activate the virtual environment:
   .\\.venv\\Scripts\\Activate.ps1
   
2. Training the models(trains all CNN´s, evaluates on MNIST, saves model weights, loss history and comparison plots):
   python main.py
   
3. Test single image(replace x with desired model):
   python main.py predict --model x --image .\\Different\_test\_data\\8\_example.png
   
4. Test all images in a folder(replace x with desired model):
   python main.py predict --model x --folder .\\Different\_test\_data
   
5. Compare ALL models on the same dataset:
   python main.py predict --models all --folder .\\Different\_test\_data
   
6. Save predictions to a CSV file(replace x with the model you tested or "all"):
   python main.py predict --model x --folder .\\Different\_test\_data --out results.csv
   

From the UI(click the run\_app.bat file, windows only):



Model selection

-Choose one of the 10 trained models



-Each model corresponds to a different number of hidden layers



-Single image testing





Upload a single handwritten digit image



View:



-predicted digit



-confidence score



-top-3 predictions



-the processed 28×28 image fed into the model





Dataset testing



Upload:



-multiple images at once or



-a ZIP file containing an entire dataset



single model mode:



-Automatically compute accuracy



-View results in a table



-Download predictions as a CSV file





Model comparison:



-Run all 10 models on the same dataset



Display:



-accuracy per model



-confidence statistics



-comparison plot of accuracy vs. model depth







**Requirements**

-Python 3.10+

-Pytorch

-Torchvision

-Streamlit

-NumPy

-Pillow





Author: Lukas Binau

Group members: Nikolaj and Christian







