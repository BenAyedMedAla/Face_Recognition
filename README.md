# Facial Emotion Recognition with Deep CNN and Bidirectional LSTM

This Jupyter notebook implements a deep learning model for facial emotion recognition using a combination of Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (LSTM) networks. The model is trained on the CK+48 dataset to classify facial expressions into 5 different emotions.

## Overview

The system uses a novel architecture that combines spatial feature extraction through deep CNNs with temporal sequence modeling using bidirectional LSTMs. This approach allows the model to capture both spatial patterns in facial expressions and temporal dynamics across image sequences.

## Dataset

- **Dataset**: CK+48 (Extended Cohn-Kanade Dataset)
- **Emotions**: 5 classes - Happy, Surprise, Anger, Sadness, Fear
- **Image Size**: 48x48 pixels (grayscale)
- **Sequence Length**: 3 frames per sample
- **Total Samples**: 250 sequences

### Dataset Distribution:
- Surprise: 83 samples
- Happy: 69 samples
- Anger: 45 samples
- Sadness: 28 samples
- Fear: 25 samples

## Model Architecture

### Deep CNN Feature Extractor
- **Input**: 48x48x1 grayscale images
- **Layers**: 8 convolutional layers with batch normalization
- **Filters**: 64 → 64 → 128 → 128 → 256 → 256 → 512 → 512
- **Activation**: ELU (Exponential Linear Unit)
- **Pooling**: MaxPooling2D after each pair of conv layers
- **Regularization**: Dropout (0.4-0.45) and BatchNormalization
- **Output**: Global max pooled features

### Sequence Model
- **TimeDistributed Layer**: Applies CNN to each frame in sequence
- **Bidirectional LSTM 1**: 128 units, returns sequences
- **Bidirectional LSTM 2**: 64 units, returns single output
- **Dense Layers**: 128 units → 5 classes (softmax)
- **Regularization**: Dropout (0.35, 0.45, 0.7)

## Requirements

```python
tensorflow>=2.0
keras
opencv-python
numpy
matplotlib
seaborn
scikit-learn
```

## Requirements

Install the required packages in your Jupyter environment:

```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn
```

## Notebook Workflow

The notebook is structured in the following sections:

1. **Data Loading & Exploration**: Load images, explore dataset distribution
2. **Data Preprocessing**: Image normalization, sequence organization, train-test split
3. **Data Visualization**: Display sample images from each emotion class
4. **Model Architecture**: Define CNN feature extractor and BiLSTM sequence model
5. **Model Training**: Train with callbacks and monitor performance
6. **Results Analysis**: Generate training curves, confusion matrix, and sample predictions
7. **Model Saving**: Export trained model for future use

## Usage

### Data Preparation
Ensure your CK+48 dataset is organized as follows:
```
CK+48/
├── happy/
├── surprise/
├── anger/
├── sadness/
└── fear/
```

### Running the Notebook
1. Update the `INPUT_PATH` variable to point to your dataset directory
2. Execute all cells in the Jupyter notebook sequentially
3. The notebook will automatically handle data loading, preprocessing, model training, and evaluation

### Model Configuration
- **Batch Size**: 32
- **Epochs**: 100
- **Optimizer**: Nadam (learning rate: 0.001)
- **Loss Function**: Categorical Crossentropy
- **Train/Validation Split**: 70/30

## Training Features

### Callbacks
- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Monitors validation accuracy
  - Factor: 0.8
  - Patience: 7 epochs
  - Minimum LR: 1e-7

### Data Preprocessing
- Image normalization (pixel values scaled to [0,1])
- Categorical encoding for labels
- Stratified train-validation split
- Sequence sorting by frame order

## Results and Visualization

The model generates several visualizations:

1. **Training History**: Accuracy and loss curves for training and validation
2. **Confusion Matrix**: Classification performance across all emotion classes
3. **Sample Predictions**: Visual comparison of true vs predicted emotions
4. **Classification Report**: Detailed precision, recall, and F1-scores

All visualizations are automatically saved as PNG files:
- `epoch_history_dcnn.png`
- `confusion_matrix_dcnn.png`
- `sample_predictions.png`

## Model Output

The trained model is saved as `DCNN_model.h5` and can be loaded for inference:

```python
from tensorflow.keras.models import load_model
model = load_model('DCNN_model.h5')
```

## Key Features

- **Sequential Processing**: Handles 3-frame sequences for temporal emotion dynamics
- **Deep Architecture**: 8 convolutional layers for rich feature extraction
- **Bidirectional Processing**: Captures both forward and backward temporal dependencies
- **Regularization**: Multiple dropout layers and batch normalization prevent overfitting
- **Comprehensive Evaluation**: Detailed metrics and visualizations

## Performance Monitoring

The training process includes:
- Real-time accuracy and loss tracking
- Validation performance monitoring
- Automatic learning rate adjustment
- Early stopping capabilities (can be added)

## File Structure

```
project/
├── facial_emotion_recognition.ipynb  # Main Jupyter notebook
├── DCNN_model.h5                    # Saved model
├── epoch_history_dcnn.png           # Training curves
├── confusion_matrix_dcnn.png        # Classification matrix
├── sample_predictions.png           # Prediction examples
└── README.md                        # This file
```

## Future Improvements

- Add data augmentation techniques
- Implement cross-validation
- Experiment with attention mechanisms
- Add real-time webcam inference
- Support for additional emotion classes
- Model quantization for deployment

## License

This project is available under the MIT License.

## Citation

If you use this code in your research, please cite the original CK+ dataset:
```
Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). 
The extended cohn-kanade dataset (ck+): A complete dataset for action unit and 
emotion-specified expression. In 2010 ieee computer society conference on computer 
vision and pattern recognition-workshops (pp. 94-101). IEEE.
```
