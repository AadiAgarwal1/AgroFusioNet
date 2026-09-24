# 🌾 AgroFusionNet

A multimodal deep learning-powered Streamlit application for agricultural analysis, designed to support crop yield prediction and crop-stress assessment using rainfall, climate, soil, vegetation (NDVI), temporal, and district-level information.

## Features

- **Multimodal Agricultural Analysis**: Combines multiple agricultural data sources:
  - Rainfall and climate data
  - Soil and nutrient information
  - NDVI-based vegetation information
  - Historical and temporal agricultural features
  - District-level information

- **Deep Learning Model**:
  - LSTM-based processing for temporal rainfall and NDVI information
  - Dedicated neural-network branches for climate and soil features
  - Cross-modal attention for feature fusion
  - Gated feature fusion
  - District embeddings
  - Separate prediction heads for yield and crop stress

- **Feature Engineering**:
  - Soil nutrient ratios and NPK balance
  - Rainfall variability and seasonal rainfall features
  - NDVI trends and temporal features
  - Rainfall shock indicators
  - Yield stability features
  - Climate-soil interaction features

- **Interactive Web Application**:
  - Built with Streamlit
  - Interactive agricultural inputs
  - Yield and stress analysis
  - Agricultural condition insights
  - Recommendation and risk-analysis components

## Setup Instructions

### 1. Clone or Download the Project

```bash
git clone https://github.com/AadiAgarwal1/AgroFusionNet.git
cd AgroFusionNet
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Required Data and Model Files

The application requires the project's datasets and trained model artifacts.

These files are intentionally not included in the public repository because they are private project resources.

Place the required files in the expected project directories before running the application.

### 5. Launch the Application

```bash
streamlit run streamlit_agrofusionnet_app.py
```

The application will normally open at:

```text
http://localhost:8501/
```

## Usage

1. Launch the Streamlit application.
2. Select or provide the required agricultural information.
3. Enter the relevant soil and environmental parameters.
4. Run the analysis.
5. Review the generated agricultural predictions, stress information, and recommendations.

## Model Overview

AgroFusionNet processes different agricultural modalities through dedicated feature-processing branches before combining them into a shared representation.

The model uses:

### Rainfall Processing

Monthly rainfall information is processed using an LSTM to capture temporal patterns.

### NDVI Processing

NDVI-related temporal information is processed using an LSTM to capture vegetation trends.

### Climate Features

Climate variables are processed through fully connected neural-network layers.

### Soil Features

Soil and nutrient variables are processed through a dedicated neural-network branch, including engineered nutrient relationships.

### Feature Fusion

The processed modalities are combined using:

- Cross-modal attention
- Gated feature fusion
- Fully connected fusion layers

### Prediction

The fused representation is used for agricultural prediction tasks including:

- Crop yield prediction
- Crop-stress assessment

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Requirements

- Python 3.10+
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

See `requirements.txt` for the complete dependency list.

## Data Privacy

Datasets, trained model files, model artifacts, and other confidential project resources are excluded from the public GitHub repository.

Please do not commit private datasets, credentials, environment files, or confidential model artifacts.

## Author

**Aadi Agarwal**

B.Tech Computer Science and Engineering  
SRM Institute of Science and Technology

---

**Built for intelligent and data-driven agricultural analysis. 🌱**
