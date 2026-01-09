# unifiedCXR-Unified-Multi-Disease-Chest-X-Ray-Diagnosis-System
A unified, modular deep-learning system for automated detection of multiple thoracic diseases from Chest X-ray images, integrating three independent CNN models into a single web-based diagnostic dashboard with real-time inference and explainable predictions. 

This project demonstrates end-to-end ML system design, from dataset exploration and preprocessing to model training, evaluation, and deployment.

Tuberculosis (TB): Kaggle – TB Chest X-Ray Database (Tawsifur Rahman)
Split:
Train: 7,039
Validation: 782
Test: 840

Pneumonia: Kaggle – Chest X-Ray Images (Pneumonia) (Paul Timothy Mooney)
Split:
Train: 5,216
Validation: 16
Test: 624

To run this:
Use Windows Powershell, after downloading the project
# 1. Go to your project folder

# 2. Create virtual environment
python -m venv mediscan_env

# 3. Activate it
mediscan_env\Scripts\activate

# 4. Install packages
pip install flask==2.3.3
pip install flask-cors==4.0.0
pip install tensorflow==2.13.0
pip install numpy==1.24.3
pip install pillow==10.0.0
pip install opencv-python==4.8.1.78
pip install python-dotenv==1.0.0

# 5. Run the application
python app.py

Cardiomegaly: NIH Chest X-Ray14 Dataset
Subset Used: Cardiomegaly: 2,776 Normal: 60,361
Split:
Train: 45,327
Validation: 8,122
Test: 9,688
