# Automated Sensitive Information Detection

This repository contains the code implementation for the paper titled "Automated detection of unstructured context-dependent sensitive information using deep learning." The project focuses on developing an automated system for detecting sensitive information in unstructured text data, leveraging deep learning techniques.

## Description

The goal of this project is to create a deep learning model capable of identifying and classifying sensitive information within unstructured text. The model utilizes state-of-the-art deep learning algorithms to analyze the contextual dependencies of the data and accurately identify sensitive information. The code implementation provided in this repository serves as a reference for replicating the experiments and methodology presented in the associated paper.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

1. Clone the repository to your local machine.

   ```shell
   git clone https://github.com/your-username/automated-sensitive-info-detection.git

2. Create and activate a virtual environment (optional but recommended).

    ```shell
    python -m venv env
    source env/bin/activate  # for Linux/Mac
    env\Scripts\activate  # for Windows

3. Install the required libraries.

    ````shell
    pip install -r requirements.txt

## Project Structure
The project structure is organized as follows:

```shell
    automated-sensitive-info-detection/
      ├── data/
      │   └── dataset.csv
      ├── src/
      │   ├── main.py
      │   ├── preprocessing.py
      │   ├── deep_learning_model.py
      │   └── evaluation.py

