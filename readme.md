
# 📘 LeNet-5 on CIFAR-10  

## 📖 Project Overview  
This project implements the **classic LeNet-5 architecture** on the **CIFAR-10 dataset** using PyTorch.  

LeNet-5 was originally designed for **handwritten digit recognition (MNIST, grayscale images)**.  
Here, the goal was not to achieve state-of-the-art performance, but to **explore its limitations** when applied to a more challenging dataset with **RGB natural images (CIFAR-10)**.  

🔑 Key idea: Even though modern CNNs (ResNet, VGG, etc.) perform far better on CIFAR-10, we restrict ourselves to LeNet-5 to understand:  
- How an early CNN behaves on color image datasets  
- Why architectural advances were necessary  
- The performance bottlenecks of small models on complex data  

---

## 🗂 Repository Structure  

```
├── notebooks/
│   └── lenet5_cifar10.ipynb   # Training & evaluation notebook
├── results/
│   ├── loss_curve.png         # Training/Validation loss curves
│   ├── accuracy_curve.png     # Training/Validation accuracy curves
│   └── training_logs.txt      # Full epoch-wise logs
├── requirements.txt           # Project dependencies
└── README.md

```

---

## 🏗 Model Architecture  

The **classic LeNet-5** consists of:  
- 2 convolutional layers  
- 2 subsampling (average pooling) layers  
- 2 fully connected layers  
- 1 output layer  

> ⚠️ No BatchNorm, No ReLU (only Tanh/Sigmoid in original paper).  
This constraint makes it hard for the model to generalize well on RGB datasets.  

---

## ⚙️ Training Details  

- **Optimizer**: SGD (momentum=0.9)  
- **Learning Rate Scheduler**: StepLR  
- **Batch Size**: 64  
- **Epochs**: 30  
- **Augmentation**: RandomCrop(32, padding=4), RandomHorizontalFlip  
- **Loss Function**: CrossEntropyLoss  

---

## 📊 Results  

### 📉 Training Logs (Summary)  
```
Epoch [1/30] | Train Acc: 18.9% | Val Acc: 24.8%
Epoch [5/30] | Train Acc: 32.7% | Val Acc: 36.6%
Epoch [10/30] | Train Acc: 38.0% | Val Acc: 41.9%
Epoch [15/30] | Train Acc: 43.3% | Val Acc: 47.7%
Epoch [20/30] | Train Acc: 46.2% | Val Acc: 49.5%
Epoch [30/30] | Train Acc: 47.6% | Val Acc: 50.6%
```

📌 **Final Validation Accuracy: ~50%**  
📌 The model plateaued around this range and could not go further.  

---

### 📈 Curves  

#### 🔹 Loss Curve  
![Loss Curve](results/loss_curve (1).png)  

#### 🔹 Accuracy Curve  
![Accuracy Curve](results/accuracy_curve (1).png)  

---

## 🔍 Key Insights  

- LeNet-5 performs **reasonably well on grayscale/simple images (MNIST)**, but struggles with complex, colored CIFAR-10 images.  
- Validation accuracy **stabilized at ~50%**, showing its **limited capacity**.  
- Heavy data augmentation actually **hurt performance**, as the shallow architecture couldn’t adapt.  
- This experiment highlights **why deeper networks (AlexNet, VGG, ResNet)** were necessary for progress in computer vision.  

---

## 🚀 Next Steps  

- Compare with a **custom CNN** of similar size but with ReLU + BatchNorm.  
- Try **transfer learning (ResNet/VGG)** to observe the gap in performance.  
- Extend this study to **grayscale CIFAR-10 (converted)** to confirm LeNet-5’s strengths.  

---

## 📌 Conclusion  

This project demonstrates that:  
✔️ **LeNet-5 is historically important** but limited in modern settings.  
✔️ It achieves ~50% accuracy on CIFAR-10, far below modern baselines.  
✔️ The exercise helps us **understand CNN evolution** and why architectural innovations matter.  

---

✨ *Learning takeaway: LeNet-5 is excellent for understanding CNN fundamentals, but unsuitable for complex RGB datasets without modifications.*  
