# computer-vision-project

This project is a deep learning pipeline for coin classification using TensorFlow/Keras. It covers data preparation, augmentation, model training, evaluation, and visualization.

## Project Structure

- `dataset/`  
  - `train/`, `validation/`, `test/` folders with subfolders for each coin class (e.g., dimes).
- `training_model/test.ipynb`  
  Main Jupyter notebook for data processing, model training, and evaluation.

## Workflow

### 1. Data Preparation & Augmentation
- Counts images in each class for train, validation, and test splits.
- Uses `ImageDataGenerator` for strong and light augmentations to balance datasets.
- Augmented images are saved to appropriate directories.

### 2. Data Generators
- Sets up Keras `ImageDataGenerator` for train, validation, and test sets.
- Loads images in batches for efficient training.

### 3. Model Architecture
- Sequential CNN with:
  - Two Conv2D + MaxPooling layers
  - Flatten and Dense layers
  - Dropout for regularization
  - Softmax output for multi-class classification

### 4. Training
- Trains the model with early stopping and model checkpointing.
- Plots training/validation accuracy and loss curves.

### 5. Evaluation
- Evaluates the model on the test set.
- Displays a confusion matrix and classification report.
- Shows sample predictions with actual and predicted labels.

## Requirements

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- matplotlib
- seaborn

Install dependencies with:
```bash
pip install tensorflow scikit-learn matplotlib seaborn
```

## Usage

1. Place your dataset in the `dataset/` folder with the structure:
    ```
    dataset/
      train/
        class1/
        class2/
        ...
      validation/
        class1/
        class2/
        ...
      test/
        class1/
        class2/
        ...
    ```
2. Open `training_model/test.ipynb` in Jupyter or VS Code.
3. Run each cell in order to preprocess data, train, and evaluate the model.

## Results

- The notebook outputs accuracy, loss plots, confusion matrix, and classification report.
- Best model is saved as `best_coin_model.h5`.

---
