# Spoken Digit Recognition using ANN

This project recognizes spoken digits (0-9) from short audio recordings
using an Artificial Neural Network.

## How it works
1. Load WAV audio (8 kHz) from the Free Spoken Digit Dataset (3000 files)
2. Extract 20 MFCC coefficients per frame using librosa
3. Summarize each audio as 40 features (mean + standard deviation)
4. Scale features with StandardScaler
5. Train a Dense ANN: 40 -> 128 -> 64 -> 10 (ReLU, Dropout 0.3, Softmax)
6. Predict the digit with a confidence score

## Results
- Test accuracy: about 97.7% (586/600)
- Total parameters: 14,154
- Early stopping used to prevent overfitting

## Files
- `spoken_digit_ann.py` - full training and evaluation code
- `digit_model.keras` - trained model
- `training_curves.png` - accuracy and loss graphs
- `confusion_matrix.png` - test set confusion matrix

## Run
pip install numpy librosa scikit-learn tensorflow matplotlib
python spoken_digit_ann.py
