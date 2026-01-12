# AWSSageMaker

**Mobile Price Classification using AWS SageMaker**

This project demonstrates how to build, train, and deploy a machine learning model using **Amazon SageMaker AI**. The repository contains a dataset for mobile price classification, Python training scripts, and a Jupyter research notebook illustrating the end-to-end workflow.

---

## 🚀 Overview

The goal of this project is to classify mobile devices into pricing categories using a machine learning model. It uses:

- **Dataset files** (`mob_price_classification_train.csv`, `train-V-1.csv`, `test-V-1.csv`)
- **Training script** (`script.py`)
- **Exploratory research** (`research.ipynb`)
- **AWS SageMaker for scalable training and deployment**

Amazon SageMaker is a fully-managed service that enables developers and data scientists to build, train, and deploy ML models at scale.

---

## 📁 Repository Structure
```
.
├── mob_price_classification_train.csv # Training dataset
├── train-V-1.csv # Train fold (if splitting manually)
├── test-V-1.csv # Test fold
├── research.ipynb # Notebook with EDA & model experiments
├── script.py # Training & inference script
├── requirements.txt # Python dependencies
├── LICENSE # MIT license
├── .gitignore
└── README.md # This file
```


---

## 🛠 Prerequisites

To work locally you’ll need:

- Python 3.8+
- AWS account with IAM permissions for SageMaker and S3
- AWS CLI configured (`aws configure`)
- SageMaker Python SDK (`sagemaker`)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## 🧠 Local Exploration
You can launch the notebook to explore the dataset and model logic:

```bash
jupyter notebook research.ipynb
```
This notebook walks through:
- Data inspection
- Feature exploration
- Baseline modeling
- Initial evaluation

## 📊 SageMaker Training & Deployment
Below is a general guide to run your ML training and deploy the model using AWS SageMaker’s managed services.

## 📌 Step 1 — Upload Data to S3
Upload your training and testing datasets to an S3 bucket:
```bash
aws s3 cp mob_price_classification_train.csv s3://<your-bucket-name>/
aws s3 cp train-V-1.csv s3://<your-bucket-name>/
aws s3 cp test-V-1.csv s3://<your-bucket-name>/
```
## 📌 Step 2 — Define SageMaker Training Job
Use the script.py as your entry point training script, containing training logic. SageMaker Training Jobs will run this script inside a managed container.
In Python:
```python
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

sess = sagemaker.Session()
role = "<your-sagemaker-execution-role>"

sklearn_estimator = SKLearn(
    entry_point="script.py",
    role=role,
    instance_type="ml.m5.large",
    framework_version="0.23-1"
)

sklearn_estimator.fit({
    "train": "s3://<your-bucket-name>/train-V-1.csv",
    "test": "s3://<your-bucket-name>/test-V-1.csv",
})
```
This will start a managed SageMaker training job where AWS handles infrastructure provisioning, scaling, and execution.

## 📌 Step 3 — Deploy the Model
After training completes, deploy the model as a real-time endpoint:

```python
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)
```
You can then send JSON requests for inference.

## 📌 Step 4 — Test the Endpoint
```python
response = predictor.predict({
    "features": [0.2, 1.5, 3.4, ..., 6.7]
})
print("Predicted class:", response)
```
This will return your model’s predicted pricing category in real-time.

## 🧪 Notes
- SageMaker will automatically scale training infrastructure and provide logs in CloudWatch.
- The SageMaker Python SDK abstracts away low-level cloud API calls.
- You can use SageMaker Studio or notebook instances to run experimentation interactively. 


## 📈 Benefits of AWS SageMaker Deployment
|Feature               |	Advantage |
|----------------------|-------------|
|**Managed Infrastructure**|	No servers to manage |
|**Autoscaling** |	Scales with traffic |
|**Monitoring & Logging** |	Integrated with CloudWatch |
|**Secure Execution** |	IAM roles & VPC options |
|**Reproducibility** |	Standardized training jobs |

Because SageMaker is designed to handle the complete ML lifecycle, it simplifies productionizing models without worrying about underlying servers or scaling issues. 
Wikipedia

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!
Feel free to fork and submit a pull request.

## 📜 License
This project is licensed under the MIT License.
