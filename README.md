# CIFAR-10 Image Classification with Convolutional Neural Networks (CNNs)

## 📖 Project Overview

This project builds and evaluates Convolutional Neural Network (CNN) models for multi-class image classification using the CIFAR-10 dataset. The dataset contains 60,000 low-resolution images (32x32x3) across 10 categories: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

Our focus is on creating models that achieve high accuracy and recall, particularly for the "deer" category, to develop AI systems for preventing deer-vehicle collisions. The enhanced model is designed for integration into future AI-powered emergency systems in vehicles.

---

## 🚀 Features

- Preprocessed CIFAR-10 dataset: normalization and one-hot encoding.
- Baseline and enhanced CNN architectures.
- Performance comparison using metrics such as accuracy, recall, and confusion matrices.
- Prediction examples with probabilities and visualizations.
- Evaluation and recommendations for further model improvements.

---

## 🗂 Dataset

- **Source**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Structure**:
  - 50,000 training images
  - 10,000 testing images
  - Categories: Airplanes, Cars, Birds, Cats, Deer, Dogs, Frogs, Horses, Ships, Trucks

---

## ⚙️ Project Workflow

### 1. **Data Preparation**
   - Loaded CIFAR-10 dataset.
   - Normalized image pixel values to [0, 1] for faster training.
   - Applied one-hot encoding to categorical target labels.

### 2. **Baseline Model**
   - **Architecture**:
     - 2 convolutional layers (64 filters each, ReLU activation, max pooling).
     - 1 dense feedforward layer with 120 neurons.
     - Output layer with 10 classes (softmax activation).
   - **Training**: Adam optimizer, learning rate = 0.001, batch size = 32, epochs = 10.
   - **Accuracy**: ~65% on the test set.

### 3. **Enhanced Model**
   - **Architecture**:
     - 3 convolutional layers (128, 64, 32 filters respectively, ReLU activation).
     - MaxPooling and dropout for regularization.
     - 2 dense feedforward layers with 128 and 64 neurons, respectively.
   - **Training**: Similar to baseline with additional dropout for regularization.
   - **Accuracy**: ~67% on the test set.
   - Improved recall for "truck" class.

### 4. **Evaluation**
   - Confusion matrix for class-wise accuracy.
   - Classification report with precision, recall, and F1-score.
   - Visualization of predictions with confidence scores.

---

## 📊 Results

- **Baseline Model**:
  - Test Accuracy: ~65%
  - Validation Loss: Slight overfitting observed.

- **Enhanced Model**:
  - Test Accuracy: ~67%
  - Recall and precision for "truck" significantly improved.
  - Better generalization compared to baseline.

---

## 📈 Visualizations

- Accuracy and loss curves for both models.
- Confusion matrix to evaluate predictions per class.
- Example predictions with true labels and confidence scores.

---

## 📚 Key Takeaways

1. **Baseline Model**: A simple CNN achieved moderate performance (~65%).
2. **Enhanced Model**: Improvements in architecture and regularization led to better accuracy (~67%) and recall.
3. **Future Work**:
   - Increase training data through augmentation.
   - Fine-tune hyperparameters (learning rate, dropout, etc.).
   - Explore transfer learning with pre-trained models like ResNet or MobileNet.

---

## 🛠️ Tools and Libraries

- Python 3.7+
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- Scikit-learn
- Plotly (optional)

---

## 👩‍💻 Contributing

Contributions are welcome! Please fork the repository and create a pull request with your proposed changes.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- CIFAR-10 Dataset by the Canadian Institute for Advanced Research.
- Inspiration from Purdue University’s study on deer-vehicle collisions.
