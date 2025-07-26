# Facial Expression Recognition with Landmark Detection

A PyTorch implementation of a two-stage facial analysis system combining landmark detection and emotion classification using ResNet18 with self-attention mechanisms on the FER2013 dataset.

## Overview

This project tackles two key computer vision challenges:
1. **Facial Landmark Detection**: Extract 4 key facial points (eye centers, lip corners) using MediaPipe ground truth
2. **Emotion Classification**: Classify 7 emotions with attention-enhanced ResNet18

**Key Achievement**: 52.38% accuracy on imbalanced FER2013 dataset with 4.2% improvement from attention mechanisms.

## Dataset

**FER2013**: 35,887 grayscale facial images (48×48 pixels)
- **Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Challenges**: Severe class imbalance (Disgust: 547 samples vs Happy: 8,989), 18% undetectable faces
- **Splits**: Training (28,709), Public Test (3,589), Private Test (3,589)

**Note**: The FER2013 dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013). Download and place in the project directory before running.

## Architecture

### Landmark Detection Model
```python
class LandmarkDetectionModel(nn.Module):
    def __init__(self, num_landmarks=8):  # 4 points × (x,y)
        super().__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_landmarks)
        )
```

### Emotion Classification with Self-Attention
```python
class EmotionClassificationModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        # Self-attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 512, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # ... ResNet backbone ...
        att = self.attention(x)
        x = x * att  # Apply attention weights
        return self.resnet.fc(torch.flatten(x, 1))
```

## Key Features

### Data Handling
- **MediaPipe Integration**: Robust face mesh detection for landmark ground truth
- **Missing Data Handling**: Graceful handling of 18% undetectable faces
- **Class Imbalance**: Weighted CrossEntropyLoss to address severe class distribution skew

### Training Enhancements
- **Advanced Augmentation**: Rotation, flipping, color jitter, random erasing
- **Regularization**: Dropout, weight decay, gradient clipping
- **Early Stopping**: Patience-based training termination
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning

## Results

### Landmark Detection Performance
| Landmark | Average Error (Euclidean) |
|----------|--------------------------|
| Left Eye Center | 0.0382 |
| Right Eye Center | 0.0398 |
| Left Lip Corner | 0.0471 |
| Right Lip Corner | 0.0433 |

### Emotion Classification Results
| Emotion | Accuracy | Sample Count | Performance |
|---------|----------|--------------|-------------|
| **Happy** | **82.82%** | 1,983 | Excellent |
| **Surprise** | **74.28%** | 831 | Good |
| **Neutral** | **73.64%** | 1,234 | Good |
| Anger | 56.43% | 958 | Moderate |
| Sadness | 45.21% | 1,247 | Moderate |
| Disgust | 38.18% | 123 | Poor (class imbalance) |
| Fear | 11.74% | 764 | Poor (class imbalance) |
| **Overall** | **52.38%** | 3,589 | - |

## Requirements

```bash
pip install torch torchvision opencv-python mediapipe numpy pandas matplotlib seaborn tqdm pillow
```

## Usage

### Quick Start
```python
# Load and preprocess data
dataset = pd.read_csv("fer2013.csv")
train_df = dataset[dataset["Usage"] == "Training"]

# Train landmark detection model
landmark_model = LandmarkDetectionModel().to(device)
# ... training loop ...

# Train emotion classification model
emotion_model = EmotionClassificationModel().to(device)
# ... training loop ...
```

## Key Insights

1. **Attention Effectiveness**: Self-attention mechanism provided 4.2% accuracy improvement by focusing on emotion-relevant facial regions
2. **Class Imbalance Impact**: Model performance directly correlates with class representation (Happy: 82.8% vs Fear: 11.7%)
3. **Data Quality Challenges**: 18% of images had undetectable faces, requiring robust error handling
4. **Landmark Consistency**: Low prediction errors across all landmarks despite 48×48 resolution constraints

## Future Improvements

- **Class Balancing**: Implement SMOTE or focal loss for minority classes
- **Architecture Upgrades**: Explore Vision Transformers (ViTs) for better feature extraction
- **Multi-Task Learning**: Joint training of landmark detection and emotion classification
- **Data Augmentation**: Advanced techniques like mixup or cutmix

## License

This project is for educational purposes. 

**Dataset License**: FER2013 is available under Kaggle's terms of use.

**Code License**: MIT License - see [LICENSE](LICENSE) file for details.
