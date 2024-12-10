{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# PySpark Data Processing and Classification\n",
    "\n",
    "This notebook demonstrates the use of **PySpark** for data processing and machine learning tasks. It includes examples of processing an e-commerce dataset and training a Decision Tree Classifier on the Iris dataset.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup and Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install pyspark\n",
    "import pyspark\n",
    "from pyspark.sql import *\n",
    "from pyspark.sql.functions import *\n",
    "from pyspark import SparkContext, SparkConf\n",
    "\n",
    "# Create the session\n",
    "conf = SparkConf().set(\"spark.ui.port\", \"4050\")\n",
    "sc = pyspark.SparkContext(conf=conf)\n",
    "spark = SparkSession.builder.getOrCreate()\n",
    "spark"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Download and Load Digikala Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!wget https://bigdata-ir.com/wp-content/uploads/2020/12/digikala_datasetwww.bigdata-ir.com_.zip\n",
    "!unzip digikala_datasetwww.bigdata-ir.com_.zip\n",
    "import os\n",
    "os.rename('digikala_dataset[www.bigdata-ir.com]', 'digikala_dataset')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load and Explore Orders Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "orders_df = spark.read.csv(\"digikala_dataset/orders.csv\" , header=True, inferSchema=True)\n",
    "orders_df.printSchema()\n",
    "orders_df.show(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Drop Unnecessary Columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "orders_df = orders_df.drop('Amount_Gross_Order', 'city_name_fa', 'Quantity_item')\n",
    "orders_df.show(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load and Process Purchase History"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "tarikhe_kharid_df = spark.read.csv(\"digikala_dataset/tarikhche kharid.csv\" , header=True)\n",
    "tarikhe_kharid_df = tarikhe_kharid_df.select('id', 'selling_price', 'product_id')\n",
    "tarikhe_kharid_df.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Join Datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "join_df = orders_df.join(tarikhe_kharid_df, orders_df.ID_Item == tarikhe_kharid_df.product_id)\n",
    "join_df.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Group and Find Popular Products"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "orders_count = join_df.groupBy('product_id').count()\n",
    "most_popular = orders_count.orderBy(desc('count')).first()\n",
    "most_popular"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Iris Dataset: Decision Tree Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyspark.ml.feature import StringIndexer, VectorAssembler\n",
    "from pyspark.ml.classification import DecisionTreeClassifier\n",
    "from pyspark.ml.evaluation import MulticlassClassificationEvaluator\n",
    "from pyspark import SparkFiles\n",
    "\n",
    "# Load the Iris dataset\n",
    "url = \"https://raw.githubusercontent.com/selva86/datasets/master/Iris.csv\"\n",
    "spark.sparkContext.addFile(url)\n",
    "df = spark.read.csv(\"file://\" + SparkFiles.get(\"Iris.csv\"), header=True, inferSchema=True)\n",
    "\n",
    "# Preprocess Data\n",
    "label_indexer = StringIndexer(inputCol=\"Species\", outputCol=\"label\")\n",
    "data = label_indexer.fit(df).transform(df)\n",
    "assembler = VectorAssembler(inputCols=[\"SepalLengthCm\", \"SepalWidthCm\", \"PetalLengthCm\", \"PetalWidthCm\"], outputCol=\"features\")\n",
    "data = assembler.transform(data)\n",
    "\n",
    "# Split Data\n",
    "train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)\n",
    "\n",
    "# Train Model\n",
    "dt_classifier = DecisionTreeClassifier(labelCol=\"label\", featuresCol=\"features\")\n",
    "model = dt_classifier.fit(train_data)\n",
    "\n",
    "# Evaluate Model\n",
    "predictions = model.transform(test_data)\n",
    "evaluator = MulticlassClassificationEvaluator(labelCol=\"label\", predictionCol=\"prediction\", metricName=\"accuracy\")\n",
    "accuracy = evaluator.evaluate(predictions)\n",
    "print(f\"Test Accuracy: {accuracy:.2f}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
