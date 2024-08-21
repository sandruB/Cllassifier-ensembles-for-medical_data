Before using the source code, please follow these steps:
 1. Python installation:
	- install python version 3.11.5 from the official website:
	https://www.python.org/downloads/release/python-3115/
	- record the installation path

 2. Set up the environment:
	- Open a terminal (Command Prompt on Windows or Terminal on macOS/Linux).
	- Navigate to the location of the Python program using:
	cd installation_path

 3. Create the virtual enviroment
	- Run the following command to create a virtual environment named env:
	python -m venv env
 
 4. Activate virtual environment
	- On Windows:
	.\env\Scripts\activate
	- On macOS/Linux:
	source env/bin/activate
	
 5. Install the dependencies:
	-With the virtual environment activated, install dependencies specified in requirements.txt:
	pip install -r requirements.txt

Some Python packages used in this project are compiled with Microsoft Visual C++ (e.g., NumPy, TensorFlow). To ensure correct execution, you may need to install Microsoft Visual C++ Redistributable. Please download it from Microsoft Visual C++ Redistributable, choosing the appropriate download link for your hardware architecture.
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

To run the program:
- Ensure that you are in the project's directory
- Run the following command in the terminal:
python main.py
 After execution, the program will display classification results based on its implementation.