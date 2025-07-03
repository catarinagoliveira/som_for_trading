# Unsupervised Clustering and Ensemble Decision Strategies in Cryptocurrency Trading  
## A SOM-Based Hybrid Model for Signal Generation


This project applies **Self-Organizing Maps (SOMs)** to generate and evaluate trading signals based on **technical indicators**, **sentiment data**, or a **hybrid** of both. It supports training, evaluation, ensemble strategies, and backtesting.
<p align="center">
  <img src="https://github.com/user-attachments/assets/4d4575d6-a339-40fe-89f7-a90892b3937c" alt="Methodological framework" width="600"/>
</p>  
<p align="center"><i>Figure 1. Methodological framework</i></p>


### Project Structure
```
.  
├── data/                  # Input data: price, sentiment  
├── ensemble/              # Strategies combining multiple SOMs  
├── evaluation/            # Evaluation metrics and backtesting tools  
├── pipeline/              # Data preprocessing and feature engineering  
├── train/                 # SOM training scripts and utilities  
├── config.py              # Global configuration  
├── main.ipynb             # Main execution notebook (demo/workflow)  
├── requirements.txt       # Required Python packages  
```

### Install dependencies: 
```
   pip install -r requirements.txt
```

Use main.ipynb to run the full pipeline, from feature selection and SOM training to signal generation and strategy evaluation.


