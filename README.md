
# MLTRANSLATOR-PY

## Project Description
This project aims to create a neural machine translation model for translating text between Croatian and English. The project encompasses data collection and preprocessing, exploratory data analysis, model selection, implementation, training, evaluation, and comparison.

## Project Structure

1. **Introduction**
    - **Problem Description**: This project addresses the challenge of machine translation between Croatian and English using neural networks. Machine translation is crucial for eliminating language barriers and enabling communication between different languages.
    - **Project Goals**: The goal is to develop a robust and efficient translation model that can accurately translate texts between Croatian and English.

2. **Data Collection and Preprocessing**
    - **Data Collection**: Data has been collected from reliable sources, including parallel corpora of Croatian and English texts (opus100, en-hr).
    - **Data Cleaning**: The data has been preprocessed to handle missing values, encode categorical variables, and normalize textual data.

3. **Feature Analysis**
    - **Exploratory Data Analysis**: Data visualization to understand the distribution of text lengths, common words, and correlations.
    - **Feature Selection**: Features such as sentence length and word frequency were selected based on the analysis.

4. **Model Selection and Implementation**
    - **Model Selection**: Several models were considered, including MBartForConditionalGeneration. The selection was based on their performance in similar translation tasks.
    - **Model Implementation**: Selected models were implemented using Hugging Face Transformers and PyTorch libraries.
    - **Cross-Validation**: Cross-validation was conducted to ensure the robustness of the model.

5. **Model Evaluation**
    - **Performance Evaluation**: Metrics such as BLEU score were used to evaluate the model's performance.
    - **Model Comparison**: The performance of different models was compared, and the best model was selected for implementation.

## Setup Instructions

### Prerequisites
- Ensure that CUDA 11.8 is installed.
- Ensure that cuDNN for CUDA 11.8 is installed.

### Setup Steps

1. **Clone the repository:**
    ```bash
    git clone https://github.com/ILISJAK/mltranslator-py.git
    cd mltranslator-py
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Check CUDA installation:**
    Ensure that CUDA is properly installed and accessible. You can verify this by running:
    ```bash
    nvcc --version
    ```
    Additionally, verify that PyTorch can access the GPU:
    ```python
    import torch
    print(torch.backends.cudnn.enabled)
    print(torch.cuda.is_available())
    ```

5. **Download and preprocess data:**
    ```bash
    python data/download_data.py
    ```

6. **Train the model:**
    ```bash
    python model/train.py
    ```

7. **Evaluate the model:**
    ```bash
    python model/evaluate.py
    ```

8. **Run the web interface:**
    ```bash
    cd web
    python app.py
    ```

9. **Open your browser and navigate to `http://127.0.0.1:5000` to use the translation interface.**

### Notes
- Ensure that you have the appropriate drivers and libraries for your GPU.
- Adjust batch sizes and datasets as needed to fit the capacity of your GPU memory.
- The evaluation script includes plotting the BLEU score to visualize model performance.

## Additional Information
- **Data Cleaning and Preprocessing**: The `data/download_data.py` script downloads and preprocesses the necessary data.
