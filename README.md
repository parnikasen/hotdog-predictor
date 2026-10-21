# Hot Dog / Not Hot Dog CNN Classifier

A Convolutional Neural Network (CNN) web application that classifies whether an image contains a hot dog or not. Features preloaded sample photos, image upload, and test evaluation metrics. No external API keys are required.

## Dataset
- **Name**: `dansbecker/hot-dog-not-hot-dog` (Kaggle)
- **Download command**:
  ```python
  import kagglehub
  path = kagglehub.dataset_download("dansbecker/hot-dog-not-hot-dog")
  print("Path to dataset files:", path)
  ```
- **Splits**:
  - Train: 498 images (249 hot dog, 249 not hot dog)
  - Test: 500 images (250 hot dog, 250 not hot dog)

## CNN Architecture
- **Input**: 64 × 64 × 3 RGB tensor
- **Stage 1**: Conv2D (16 filters, 3×3, ReLU) + MaxPooling2D (2×2)
- **Stage 2**: Conv2D (32 filters, 3×3, ReLU) + MaxPooling2D (2×2)
- **Stage 3**: Conv2D (64 filters, 3×3, ReLU) + MaxPooling2D (2×2)
- **Head**: Flatten $\rightarrow$ Dense (32 units, ReLU) $\rightarrow$ Dense (1 unit, Sigmoid)
- **Total Parameters**: 154,721
- **Optimizer**: Adam (lr = 0.0005)
- **Loss**: Binary Cross-Entropy

## Test Set Evaluation Metrics
- **Accuracy**: 86.6%
- **Precision**: 87.04%
- **Recall**: 86.0%
- **F1-Score**: 86.52%
- **Loss**: 0.3812
- **Confusion Matrix** (500 test samples):
  - True Positive (Hot Dog $\rightarrow$ Hot Dog): 215
  - False Positive (Not Hot Dog $\rightarrow$ Hot Dog): 32
  - True Negative (Not Hot Dog $\rightarrow$ Not Hot Dog): 218
  - False Negative (Hot Dog $\rightarrow$ Not Hot Dog): 35

## Running Locally

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm run dev
   ```
   Open `http://localhost:3000` in your browser.

3. **Build for production**:
   ```bash
   npm run build
   npm start
   ```

## Tech Stack
- **Backend**: Node.js, Express, `@tensorflow/tfjs`, `jimp`
- **Frontend**: React 19, TypeScript, Tailwind CSS, Lucide icons
- **Zero API Keys**: Runs local CNN inference on-server
