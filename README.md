# MLTRANSLATOR-PY

## Project Description
This project aims to create a neural machine translation model to translate text between Croatian and English.

## Setup Instructions

### Prerequisites
- Ensure you have CUDA 12.5 installed.
- Ensure you have cuDNN for CUDA 12.5 installed.

### Step-by-Step Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/ILISJAK/mltranslator-py.git
    cd mltranslator-py
    ```

2. **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Verify CUDA installation:**
    Make sure that CUDA is properly installed and accessible. You can do this by running:
    ```bash
    nvcc --version
    ```
    Additionally, verify that PyTorch can access the GPU:
    ```python
    import torch
    print(torch.backends.cudnn.enabled)
    print(torch.cuda.is_available())
    ```

5. **Download and preprocess the data:**
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

9. **Open a browser and go to `http://127.0.0.1:5000` to use the translation interface.**

### Notes
- Make sure you have the appropriate drivers and libraries for your GPU.
- Adjust batch sizes and dataset sizes as needed to fit your GPU memory constraints.
