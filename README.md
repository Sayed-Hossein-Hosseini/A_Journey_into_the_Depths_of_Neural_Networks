# ✨ A Journey into the Depths of Neural Networks: From Single Neurons to Convolutional Architectures ✨

<p align="center">
  <a href="https://colab.research.google.com/github/Sayed-Hossein-Hosseini/A_Journey_into_the_Depths_of_Neural_Networks/blob/main/YOUR_MAIN_NOTEBOOK_FOR_COLAB_LINK.ipynb"> <!-- Replace YOUR_MAIN_NOTEBOOK_FOR_COLAB_LINK.ipynb with the actual notebook if you want a direct Colab link -->
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

This repository chronicles an immersive, step-by-step expedition into the fascinating world of neural networks. Starting from the fundamental building block—the single neuron—we progressively construct and demystify complex architectures, culminating in the implementation of powerful Convolutional Neural Networks (CNNs) using PyTorch. Each Jupyter notebook herein represents a distinct milestone in this learning odyssey, meticulously detailing concepts and code. All custom neural network components are built from scratch using NumPy, while the advanced CNN segment leverages the efficiency and capabilities of PyTorch.

The **CIFAR-10 dataset** serves as the primary proving ground for all models developed throughout this project.

---

## 🗺️ Table of Contents

1.  [The Expedition's Blueprint: An Overview](#-the-expeditions-blueprint-an-overview)
2.  [Core Concepts Unveiled](#-core-concepts-unveiled)
3.  [The Toolkit: Technologies Employed](#-the-toolkit-technologies-employed)
4.  [Setting Up Your Lab: Installation Guide](#-setting-up-your-lab-installation-guide)
5.  [Launching the Experiments: Running the Notebooks](#-launching-the-experiments-running-the-notebooks)
6.  [Milestones & Discoveries: Notebook Details and Results](#-milestones--discoveries-notebook-details-and-results)
    *   [Milestone 1: A Neuron Dancing in Logistic Regression Style](#milestone-1-a-neuron-dancing-in-logistic-regression-style)
    *   [Milestone 2: Union of Neurons - A Hidden Layer Emerges (Binary Classification)](#milestone-2-union-of-neurons---a-hidden-layer-emerges-binary-classification)
    *   [Milestone 3: Conquering Complexity - Entering the Multi-Class Realm](#milestone-3-conquering-complexity---entering-the-multi-class-realm)
    *   [Milestone 4: Forging Ahead - Diverse Networks & Advanced Optimizers](#milestone-4-forging-ahead---diverse-networks--advanced-optimizers)
    *   [Milestone 5: The Deep Dive - Unleashing Convolutional Neural Networks (CNNs)](#milestone-5-the-deep-dive---unleashing-convolutional-neural-networks-cnns)
7.  [Future Explorations](#-future-explorations)
8.  [Join the Expedition: Contributing](#-join-the-expedition-contributing)
9.  [Usage Rights: License](#-usage-rights-license)
10. [Acknowledgements & Gratitude](#-acknowledgements--gratitude)

---

## 🚀 The Expedition's Blueprint: An Overview

This project unfolds as a curated sequence of Jupyter notebooks, each meticulously building upon the insights and implementations of its predecessor:

1.  **Single Neuron (Logistic Regression Demystified):**
    *   Crafting a single neuron from foundational principles using NumPy.
    *   Tackling binary classification on CIFAR-10 (e.g., Airplane vs. Not-Airplane).
    *   Mastering core mechanics: activation functions (Sigmoid), forward propagation, the art of backward propagation, and gradient descent.

2.  **Neural Network with One Hidden Layer (Binary Classification Enhanced):**
    *   Evolving the single neuron into a network featuring a single hidden layer.
    *   Continuing with binary classification on CIFAR-10.
    *   Illustrating how interconnected neurons collaborate to discern more intricate patterns.

3.  **Multi-Class Classifier (Expanding Horizons):**
    *   Adapting the custom neural network architecture for comprehensive multi-class classification (all 10 CIFAR-10 classes).
    *   Introducing the Softmax activation for the output layer and the categorical cross-entropy loss function.
    *   Implementing one-hot encoding for precise target label representation.

4.  **Exploring Optimizers and Activations (Refining the Engine):**
    *   Implementing and rigorously comparing diverse optimization algorithms (SGD, Momentum, Adam) from scratch.
    *   Investigating various activation functions (e.g., Tanh, ReLU) and sophisticated weight initialization techniques (He initialization).
    *   Application within the custom multi-class neural network framework on CIFAR-10.

5.  **Convolutional Neural Networks (CNNs) with PyTorch (Entering the Deep Learning Arena):**
    *   Transitioning from bespoke NumPy implementations to the robust PyTorch ecosystem.
    *   Constructing and training a foundational CNN for CIFAR-10 image classification.
    *   Introducing an advanced CNN architecture incorporating Batch Normalization and Dropout for superior performance and generalization.

---

## 🧠 Core Concepts Unveiled

Throughout this journey, you will gain hands-on experience and a deep understanding of:

*   **Neural Network Fundamentals:** Neurons, weights, biases, layers (input, hidden, output).
*   **Activation Functions:** Sigmoid, ReLU, Tanh, Softmax – their purpose and mathematical underpinnings.
*   **Forward & Backward Propagation:** The engine of learning in neural networks.
*   **Loss Functions:** Binary Cross-Entropy, Categorical Cross-Entropy – quantifying prediction errors.
*   **Gradient Descent & Optimizers:** SGD, Momentum, Adam – strategies for efficient learning.
*   **Weight Initialization:** Random, He Initialization – setting the stage for effective training.
*   **Data Preprocessing:** Normalization, flattening, one-hot encoding – preparing data for the network.
*   **Training Techniques:** Batching and Shuffling – for robust and efficient model training.
*   **Classification Paradigms:** Binary vs. Multi-class classification.
*   **Convolutional Neural Networks (CNNs):** Convolutional layers, pooling layers, fully connected layers – the workhorses of computer vision.
*   **PyTorch Framework:** Tensor operations, `nn.Module` for custom models, built-in optimizers, and data loaders.
*   **Regularization Techniques:** Dropout, Batch Normalization (in CNNs) – combating overfitting.
*   **Model Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix – rigorously assessing performance.

---

## 🛠️ The Toolkit: Technologies Employed

*   **Python 3.x:** The core programming language.
*   **NumPy:** For crafting custom neural network components and numerical computations from scratch.
*   **Pandas:** Utilized for data manipulation and analysis tasks.
*   **Matplotlib & Seaborn:** For creating insightful visualizations (e.g., confusion matrices, learning curves).
*   **TensorFlow/Keras:** Leveraged primarily for dataset loading (CIFAR-10) and utility functions like one-hot encoding.
*   **Scikit-learn:** For comprehensive evaluation metrics and machine learning utilities.
*   **PyTorch:** The framework of choice for implementing and training advanced Convolutional Neural Networks.
*   **Jupyter Notebook / Google Colab:** For interactive development, experimentation, and clear presentation.

---

## ⚙️ Setting Up Your Lab: Installation Guide

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Sayed-Hossein-Hosseini/A_Journey_into_the_Depths_of_Neural_Networks.git
    cd A_Journey_into_the_Depths_of_Neural_Networks
    ```

2.  **Establish a Virtual Environment (Highly Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    A `requirements.txt` file is the preferred method. Create one with the following, then install:
    ```text
    # requirements.txt
    numpy
    pandas
    matplotlib
    seaborn
    tensorflow
    scikit-learn
    torch
    torchvision
    jupyter
    ```
    Install using:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, install packages individually:
    ```bash
    pip install numpy pandas matplotlib seaborn tensorflow scikit-learn torch torchvision jupyter
    ```

4.  **GPU Configuration (Essential for PyTorch CNNs):**
    If you possess an NVIDIA GPU, ensure the CUDA toolkit and cuDNN are correctly installed. PyTorch will automatically detect and utilize an available GPU. Verify by running `torch.cuda.is_available()` in a Python interpreter.

---

## ▶️ Launching the Experiments: Running the Notebooks

1.  Activate your configured virtual environment.
2.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
3.  Navigate to the cloned repository directory within the Jupyter interface and open the desired notebooks.
4.  It is **strongly recommended** to execute the notebooks sequentially, as outlined in [The Expedition's Blueprint](#-the-expeditions-blueprint-an-overview), to follow the intended learning progression.
5.  Each notebook is designed to be self-contained and can be run independently.
6.  For the CNN notebook (`Entering_the_World_of_Deep_Neural_Networks_CNN.ipynb`), utilizing a **GPU is crucial for practical training times**. If running on Google Colab, ensure you select a GPU-accelerated runtime environment.

---

## 📊 Milestones & Discoveries: Notebook Details and Results

Below is a detailed breakdown of each notebook, its objectives, and typical performance metrics on the CIFAR-10 test set. Note that results from custom NumPy implementations may exhibit slight variations due to random initializations and stochastic training processes.

### Milestone 1: A Neuron Dancing in Logistic Regression Style

*   **Filename:** `A_Neuron_Dancing_in_Logistic_Regression_Style.ipynb`
*   **Objective:** Implement a single neuron, mirroring logistic regression, for binary classification (Airplane vs. Not-Airplane) on CIFAR-10.
*   **Key Implementations:**
    *   `Activation` class: `sigmoid` and `sigmoid_derivative`.
    *   `DenseLayer` class: Encapsulating a single layer's logic.
    *   `NeuralNetwork` class: Orchestrating training (binary cross-entropy loss) and prediction.
*   **Dataset Task:** CIFAR-10, transformed for binary classification (Airplane = 0, Other classes = 1).
*   **Typical Evaluation Metrics:**
    ```
    Confusion Matrix:
     [[ 264  736]
     [ 134 8866]]
    F1 Score: 0.9532
    Accuracy: 0.9130
    Precision: 0.9233
    Recall: 0.9851
    ```

### Milestone 2: Union of Neurons - A Hidden Layer Emerges (Binary Classification)

*   **Filename:** `Union_of_Neurons_in_a_Hidden_Layer_in_a_Binary_Style.ipynb`
*   **Objective:** Extend the single neuron concept to a neural network with one hidden layer, maintaining the binary classification task.
*   **Key Implementations:**
    *   Reuses `DenseLayer` and `NeuralNetwork` classes.
    *   Model Architecture: Input -> Dense(64 units, sigmoid activation) -> Dense(1 unit, sigmoid activation).
*   **Dataset Task:** CIFAR-10, binary classification (Airplane = 0, Other classes = 1).
*   **Typical Evaluation Metrics:**
    ```
    Confusion Matrix:
     [[ 319  681]
     [ 116 8884]]
    F1 Score: 0.9571
    Accuracy: 0.9203
    Precision: 0.9288
    Recall: 0.9871
    ```

### Milestone 3: Conquering Complexity - Entering the Multi-Class Realm

*   **Filename:** `Entering_the_World_of_Multi_Class_Classifiers.ipynb`
*   **Objective:** Engineer a custom neural network capable of classifying all 10 distinct classes of the CIFAR-10 dataset.
*   **Key Implementations:**
    *   `Activation` class augmented with `softmax`.
    *   `NeuralNetwork` class enhanced for multi-class scenarios:
        *   Categorical cross-entropy loss implementation.
        *   One-hot encoding for target labels.
        *   Derivation of gradient for softmax combined with cross-entropy.
    *   Model Architecture: Input -> Dense(64 units, sigmoid) -> Dense(10 units, softmax).
*   **Dataset Task:** CIFAR-10, full 10-class classification.
*   **Typical Evaluation Metrics:**
    ```
    Confusion Matrix:
     [[325  40  72  76 149  50  29  24 156  79]
      [ 25 474  13  57  26  42  32  22 104 205]
      [ 48  19 235 172 231 131  87  31  14  32]
      [ 10  14  46 406  99 225 116  20  20  44]
      [ 18  15  76 117 487 102  83  59  28  15]
      [  6   8  45 303  84 396  79  36  22  21]
      [  4  12  50 169 133  92 481  18  18  23]
      [ 24  19  34 118 180 151  34 369  16  55]
      [ 52  66  20  62  71  36  17   8 586  82]
      [ 23 141  11  94  34  58  31  37  76 495]]
    F1 Score (macro): 0.4289
    Accuracy: 0.4254
    Precision (macro): 0.4589
    Recall (macro): 0.4254
    ```

### Milestone 4: Forging Ahead - Diverse Networks & Advanced Optimizers

*   **Filename:** `Diversity_of_Networks_and_Optimizers_for_Progress.ipynb`
*   **Objective:** Investigate the impact of different activation functions (Tanh), weight initialization (He), and implement various optimization algorithms from scratch.
*   **Key Implementations:**
    *   `Activation` class expanded with `tanh` and `tanh_derivative`.
    *   `DenseLayer` enhanced with He Initialization.
    *   Optimizer classes implemented: `SGD`, `Momentum`, `Adam`.
    *   `NeuralNetwork` class adapted to integrate pluggable optimizer objects.
    *   Model Tested: Input -> Dense(64, sigmoid) -> Dense(10, softmax) trained with `Momentum` optimizer and He Initialization.
*   **Dataset Task:** CIFAR-10, full 10-class classification.
*   **Typical Evaluation Metrics (for Sigmoid + Momentum with He Init):**
    ```
    Confusion Matrix:
     [[451  57  80  21  66  24  27  45 184  45]
      [ 52 534  32  33  25  15  27  28  91 163]
      [ 88  21 326  82 187  84  84  77  33  18]
      [ 37  35  93 265 102 184 117  91  43  33]
      [ 52  20 118  71 459  54  79 100  34  13]
      [ 27  23  80 207 109 328  64  99  36  27]
      [ 22  28  89 106 158  70 453  36  25  13]
      [ 46  21  80  66 122  86  26 476  27  50]
      [ 98  70  26  34  42  29  13  17 616  58]
      [ 69 201  16  53  36  23  22  67  75 427]]
    F1 Score (macro): 0.4320
    Accuracy: 0.4335
    Precision (macro): 0.4339
    Recall (macro): 0.4335
    ```
    *(Note: This notebook also defines ReLU and Tanh activations, and SGD/Adam optimizers. Experimentation by modifying the main training block is encouraged to observe their effects.)*

### Milestone 5: The Deep Dive - Unleashing Convolutional Neural Networks (CNNs)

*   **Filename:** `Entering_the_World_of_Deep_Neural_Networks_CNN.ipynb`
*   **Objective:** Transition to PyTorch to implement and train sophisticated Convolutional Neural Networks (CNNs) for CIFAR-10 image classification.
*   **Key Implementations:**
    *   Efficient data loading and preprocessing using `torchvision.transforms` and `DataLoader`.
    *   A foundational `CNN` class.
    *   An `ImprovedCNN` class featuring multiple convolutional blocks, Batch Normalization, and Dropout for enhanced performance.
    *   A PyTorch-idiomatic training loop utilizing `nn.CrossEntropyLoss` and `optim.Adam`.
*   **Dataset Task:** CIFAR-10, full 10-class classification.
*   **Typical Evaluation Metrics (for `ImprovedCNN`):**
    ```
    Accuracy of the network on the 10000 test images: 86.80 %

    Accuracy per class:
    Accuracy of plane : 88.00 %
    Accuracy of car   : 96.10 %
    Accuracy of bird  : 82.70 %
    Accuracy of cat   : 73.00 %
    Accuracy of deer  : 85.80 %
    Accuracy of dog   : 81.70 %
    Accuracy of frog  : 90.40 %
    Accuracy of horse : 88.00 %
    Accuracy of ship  : 92.10 %
    Accuracy of truck : 90.20 %
    ```
    *(Note: The initial, simpler CNN structure in the notebook may yield lower accuracy if trained for the same duration, highlighting the advantages of the improved architecture.)*

---

## 🔭 Future Explorations

The journey doesn't end here! Potential future directions include:

*   Implementing more advanced CNN architectures (e.g., ResNet, VGG, EfficientNet) from scratch or leveraging PyTorch.
*   Exploring sophisticated data augmentation techniques tailored for CIFAR-10.
*   Implementing and comparing other regularization methods like L1/L2 regularization.
*   Visualizing learned features and filters within CNNs to gain deeper insights.
*   Experimenting with systematic hyperparameter tuning strategies (e.g., Grid Search, Bayesian Optimization).
*   Applying the learned concepts and developed frameworks to diverse and challenging datasets.

---

## 🤝 Join the Expedition: Contributing

Contributions, issue reports, and feature requests are warmly welcomed! Your expertise can help make this project even better.

1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Please check the [issues page](https://github.com/Sayed-Hossein-Hosseini/A_Journey_into_the_Depths_of_Neural_Networks/issues) for existing ideas or to report new ones.

---

## 📜 Usage Rights: License

This project is licensed under the MIT License. See the `LICENSE` file for full details. (It's recommended to add a `LICENSE` file with the MIT license text to your repository).

---

## 🙏 Acknowledgements & Gratitude

*   The creators and maintainers of the **CIFAR-10 dataset**.
*   The brilliant developers behind **NumPy, PyTorch, TensorFlow, Matplotlib, Seaborn, and Scikit-learn**, whose tools make such projects possible.
*   The vast wealth of **online resources, courses, and research papers** that have illuminated this learning path.
*   The original author of these notebooks: **[Sayed-Hossein-Hosseini](https://github.com/Sayed-Hossein-Hosseini)**.
*   The vibrant **open-source community** for fostering an environment of learning and collaboration.
