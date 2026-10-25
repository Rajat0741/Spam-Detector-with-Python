Spam Detector with Python

This repository provides a Python implementation to detect spam messages using machine learning. The package lives under `src/spam_detector_with_python/` and exposes a `main()` entry point. The project is suitable for learning about text classification, natural language processing, and basic ML workflows.

Features

- Loads and processes SMS message data for spam detection
- Cleans and vectorizes message text using CountVectorizer
- Trains a Multinomial Naive Bayes classifier
- Evaluates model accuracy and prints results
- Allows users to test custom messages

Requirements

Python 3.14 or higher  
Dependencies are declared in `pyproject.toml` (nltk, pandas, scikit-learn, numpy).

Setup

Install dependencies:
uv sync

Download or prepare a dataset in CSV format with columns for "label" and "message".

Example rows:
label,message
ham,Hey, are we still meeting tomorrow?
spam,You’ve won a free prize! Click this link now

Place the dataset at `data/spam_dataset.csv` (or adjust the path in `src/spam_detector_with_python/cli.py`).

Running the Spam Detector

To train and test the model, run:
uv run spam-detector-with-python

Or run the module directly:
uv run python -m spam_detector_with_python.cli

You will see output showing model accuracy, confusion matrix, and examples of predictions.

Example Output

Model accuracy: 98.5 percent
Confusion Matrix:
[[962 12]
 [15 141]]

Test examples:
Message: "Congratulations, you have won a lottery! Call now."
Prediction: spam

Message: "Can we reschedule our meeting?"
Prediction: ham

Custom Testing

You can modify the script to input your own message for prediction.  
Example code snippet:
message = ["Your free coupon is waiting. Reply now!"]
prediction = model.predict(vectorizer.transform(message))
print(prediction)

Contributing

Feel free to suggest improvements or submit pull requests. For questions or feedback, open an issue in the repository.

This README now includes setup instructions, sample data, example output, and a demonstration of custom usage. Let me know if you want even more detailed usage or code examples.
