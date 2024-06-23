# Language Translation Project

## Project Description
This project aims to create a neural machine translation model to translate text between Croatian and English.

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd language-translation-project
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download and preprocess the data:
    ```bash
    python data/download_data.py
    ```

5. Train the model:
    ```bash
    python model/train.py
    ```

6. Evaluate the model:
    ```bash
    python model/evaluate.py
    ```

7. Run the web interface:
    ```bash
    cd web
    python app.py
    ```

8. Open a browser and go to `http://127.0.0.1:5000` to use the translation interface.


