# Text Generation App

This repository contains a Streamlit web application for generating text using a pre-trained character-level recurrent neural network (CharRNN) model and fine-tuning the model based on the generated text.

## Features

- **Text Generation**: Generate text based on a provided starting text, controlling the number of characters to generate and the temperature parameter for controlling randomness.
- **Model Fine-tuning**: Fine-tune the pre-trained CharRNN model based on the generated text to adapt it to specific tasks or styles.
- **Model Management**: Upload, download, and delete model files via HTTP requests to a specified API endpoint.
- **Visualization**: Plot gradients and weight histograms to provide insights into the model training process.

## Usage

### Setup

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r req.txt`.
3. Make sure to have the pre-trained model file (`language_model.pth`) and the dataset file (`dataset.txt`) in the repository directory.

### Running the App

Run server:

```
cd server
python3 server.py
```

Run the Streamlit app using the following command:

```
cd client
streamlit run client.py
```


Access the app through the provided URL.

### Usage Instructions

1. Enter the starting text for text generation in the provided input field.
2. Adjust the sliders to control the number of characters to generate and the temperature parameter.
3. Click the "Generate Text" button to generate text based on the input.
4. Optionally, click the "Train Model with Generated Text" button to fine-tune the model based on the generated text.
5. Click the "Pull New Version" button to download a new version of the model from the server.

## File Structure

- `app.py`: Main Streamlit application script.
- `README.md`: This file, providing information about the project.
- `requirements.txt`: List of Python dependencies required for running the application.
- `dataset.txt`: Text dataset used for training the CharRNN model.
- `language_model.pth`: Pre-trained CharRNN model parameters file.
- `server_model.pth`: Model file to be uploaded to the server.
- `downloaded_model.pth`: Downloaded model file from the server.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests for any improvements or fixes.
