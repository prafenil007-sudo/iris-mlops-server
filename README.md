\# Iris MLOps Classifier



A Flask web application and JSON API that predicts Iris flower species from four measurements. The model is packaged in a Docker container and deployed on AWS EC2.



\## AWS deployment



Application URL: http://3.108.236.199:5000



Health check: http://3.108.236.199:5000/health



\## Run tests



```cmd

.\\venv\\Scripts\\python.exe -m pytest -v

