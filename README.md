# PySpark Data Processing and Classification

This project demonstrates the use of **PySpark** for data processing and machine learning tasks. It includes examples of working with an e-commerce dataset (Digikala) and training a machine learning model (Decision Tree Classifier) on the Iris dataset.

---

## Features

1. **Data Processing with PySpark**:
   - Load and preprocess datasets.
   - Perform schema inference and transformations.
   - Join datasets for combined analysis.
   - Generate new features based on conditions and calculations.

2. **Machine Learning**:
   - Train a **Decision Tree Classifier** on the Iris dataset.
   - Evaluate the model's performance using accuracy metrics.

---

## Getting Started

### Prerequisites

Before running the project, make sure you have:

- Python 3.x
- Jupyter Notebook
- PySpark (install via pip: `pip install pyspark`)

---

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Parsa-Jafargholi/PySpark-Data-Processing.git
   cd PySpark-Data-Processing
   ```

2. Install dependencies:
   ```
   pip install pyspark
   ```

3. Open the Jupyter Notebook:
   ```
   jupyter notebook
   ```

---

## Dataset Information

### 1. **Digikala Dataset**:
   - **Source**: [BigData-IR](https://bigdata-ir.com/)
   - **Description**: This dataset contains e-commerce transaction data, including orders, product details, and purchase history.

### 2. **Iris Dataset**:
   - **Source**: [Selva86 Datasets](https://github.com/selva86/datasets)
   - **Description**: A well-known dataset for multi-class classification tasks, containing 150 samples of three species of Iris flowers.

---

## How to Use

1. Open the notebook:
   - [PySpark Data Processing and Classification.ipynb](https://github.com/Parsa-Jafargholi/PySpark-Data-Processing/blob/main/PySpark%20Data%20Processing%20and%20Classification.ipynb)

2. Follow the steps in the notebook:
   - Data Loading and Preprocessing
   - Feature Engineering
   - Model Training and Evaluation

---

## Results

- **Digikala Dataset**:
  - The most popular product: **Product ID 179064**, with **34,095 orders**.

- **Iris Dataset**:
  - Decision Tree Classifier accuracy: **92%**.

---

## Project Structure

```
PySpark-Data-Processing/
│
├── PySpark Data Processing and Classification.ipynb  # Main notebook
├── README.md                                        # Project documentation
└── digikala_dataset/                                # Dataset folder (optional for local use)
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any questions or feedback, feel free to contact:

- **GitHub**: [Parsa Jafargholi](https://github.com/Parsa-Jafargholi)
- **Email**: your_email@example.com
