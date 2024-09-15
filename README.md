
# CO:CO2 Ratio Prediction App

## Overview

The **CO:CO2 Ratio Prediction App** is an interactive web application built using Streamlit and machine learning techniques. It allows users to predict the CO:CO2 ratio based on various environmental parameters. Whether you're interested in assessing combustion efficiency or understanding environmental impact, this app provides real-time predictions.

## Features

- **User-Friendly Interface**: The app offers an intuitive interface where users can input environmental parameters relevant to their scenario.

- **Machine Learning Model**: At its core, the app utilizes a trained machine learning model. This model has been fine-tuned using historical data to predict the CO:CO2 ratio accurately.

- **Real-Time Predictions**: Users receive immediate feedback on the predicted CO:CO2 ratio based on their input parameters. This dynamic interaction enhances usability.

## Getting Started

1. **Installation**:
   - Clone this repository to your local machine.
   - Install the necessary dependencies using `pip install -r requirements.txt`.

2. **Run the App**:
   - Navigate to the project directory.
   - Execute `streamlit run co_co2_predict_app.py` in your terminal.
   - The app will open in your default web browser.

3. **Input Parameters**:
   - Enter relevant environmental parameters (e.g., CB_FLOW, CB_PRESS, STEAM_FLOW, etc.) through the user-friendly interface.

4. **View Predictions**:
   - Click the "PREDICT" button to obtain the predicted CO:CO2 ratio.

## Example Usage

![Example Usage](example_usage.gif)

## Acknowledgments

- The dataset used for training the model (bf3_co_co2_input.csv) was sourced from [XYZ Dataset Repository](link-to-dataset).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to submit a pull request.

