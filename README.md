# Gambling Participation Prediction
## Overview

This project evaluates how well demographic and socioeconomic data from the **American Community Survey (ACS)** predict **gambling participation rates by ZIP code**. The primary objective is to compare predictive performance across multiple modeling approaches.

Models evaluated:

* **LASSO regression**
* **Random Forest (RF)**
* **XGBoost**

---

## Data

* **Outcome**: ZIP-code–level online gambling participation rate
* **Predictors**: ACS demographic and socioeconomic characteristics (e.g., sex, race/ethnicity, education, income, poverty, disability, household composition)

---

## Methods

* All models are evaluated using **cross-validation** and test-set performance to find appropriate hyperparameter settings

---

## Evaluation Metrics

* **RMSE**
* **R2**

---

## Goal

To assess trade-offs between **interpretability** (LASSO) and **predictive performance** (RF, XGBoost) when modeling gambling participation using aggregated ACS data.

---

## Note
Data provided excludes gambling data as it is proprietary.
