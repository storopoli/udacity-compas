# COMPAS Fair Classifier

This is a Capstone Project for the Udacity's [Machine Learning Engineer nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t). The goal is to **train/tune/deploy a fair binary classifier** for recidivism using [COMPAS data](https://github.com/propublica/compas-analysis).

## COMPAS 2016 Scandal

In May 2016, [ProPublica published a report](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) regarding a judicial decision support algorithm that outputs a risk score for defendant redicivism. This algorithm is called COMPAS, short for *Correctional Offender Management Profiling for Alternative Sanctions*. It has been shown that COMPAS is extremely biased toward african-american offenders when compared to caucasian offenders for the same prior/post offenses. 

## Model

The model employed was [SageMaker's `XGBoost`](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html) tunned to maximize MAP (mean average precision) in order to deal with original COMPAS' model unbalanced false positive rates.

## Results

### Original COMPAS

|                | **African-American** | **Caucasian** |
| -------------- | -------------------- | ------------- |
| **Accuracy**   |                      |               |
| False Positive |                      |               |

### Proposed Model

|                    | **African-American** | **Caucasian** |
| ------------------ | -------------------- | ------------- |
| **Accuracy**       |                      |               |
| **False Positive** |                      |               |

## Author

 Jose Storopoli, PhD - [ORCID](https://orcid.org/0000-0002-0559-5176) - [CV](https://storopoli.github.io)

[thestoropoli@gmail.com](mailto:thestoropoli@gmail.com)