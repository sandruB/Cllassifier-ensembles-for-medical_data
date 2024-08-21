# Classification of Selected Medical Data with use of Classifier Ensembles
## Project Overview
This project aims to develop an ensemble classifier suitable for the classification of medical data, that would achieve better results than individual (base) classifiers. The research involves three datasets: Breast Cancer Wisconsin (Diagnostic), Heart Failure Prediction, and Stroke Prediction. Various base classifiers, including Logistic Regression, Support Vector Machine, Random Forest, Extra Trees Classifier, K-Nearest Neighbors, Naive Bayes, Gradient Boosting, Decision Tree, and Multilayer Perceptron, were considered and their parameters were searched using grid search approach. Soft Voting was used as an ensemble method and all possible combinations of at least three base classifiers were examined. The classifiers were evaluated using performance metrics such as accuracy, precision, recall, and F1-score, which was the main metric used to assess classification quality.
## Files in Repository 
- main.py - .py file containing the complete source code of the project
- ensemble_classification_of_selected_medical_data.ipynb - Jupyter Notebook with complete source code of the project
- breast_cancer.csv - Breast Cancer Wisconsin (Diagnostic) dataset
- heart.csv - Heart Failure Prediction dataset
- stroke.csv - Stroke Prediction dataset
- requirements.txt - list of libraries and their versions needed for the project
##Before using the source code, please follow these steps:
### 1. Python installation:
	- install python version 3.11.5 from the official website:
	https://www.python.org/downloads/release/python-3115/
	- record the installation path

 ### 2. Set up the environment:
	- Open a terminal (Command Prompt on Windows or Terminal on macOS/Linux).
	- Navigate to the location of the Python program using:
	cd installation_path

### 3. Create the virtual enviroment
	- Run the following command to create a virtual environment named env:
	python -m venv env
 
 ### 4. Activate virtual environment
	- On Windows:
	.\env\Scripts\activate
	- On macOS/Linux:
	source env/bin/activate
	
 ### 5. Install the dependencies:
	-With the virtual environment activated, install dependencies specified in requirements.txt:
	pip install -r requirements.txt

Some Python packages used in this project are compiled with Microsoft Visual C++ (e.g., NumPy, TensorFlow). To ensure correct execution, you may need to install Microsoft Visual C++ Redistributable. Please download it from Microsoft Visual C++ Redistributable, choosing the appropriate download link for your hardware architecture.
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

To run the program:
- Ensure that you are in the project's directory
- Run the following command in the terminal:
python main.py
 After execution, the program will display classification results based on its implementation.

## Acknowledgments
Thank you to the creators of the Loans and Liability dataset and the contributors to the libraries used in this project.
