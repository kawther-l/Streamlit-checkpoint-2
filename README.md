# Financial Inclusion in Africa - Prediction Model

## Project Overview

In this checkpoint, we work on the **Financial Inclusion in Africa** dataset provided by the Zindi platform.

The dataset contains demographic information and financial service usage data for approximately **33,600 individuals** across East Africa.

The goal of this machine learning model is to predict which individuals are most likely to have or use a bank account.

Financial inclusion means that individuals and businesses have access to useful and affordable financial products and services that meet their needs — including transactions, payments, savings, credit, and insurance — delivered in a responsible and sustainable way.

---

## Dataset Description

| Variable                | Description                                                                                         |
|-------------------------|-------------------------------------------------------------------------------------------------|
| **country**             | Country where the interviewee is located                                                         |
| **year**                | Year the survey was conducted                                                                     |
| **uniqueid**            | Unique identifier for each interviewee                                                           |
| **location_type**       | Type of location: Rural or Urban                                                                  |
| **cellphone_access**    | If interviewee has access to a cellphone: Yes or No                                              |
| **household_size**      | Number of people living in the same household                                                    |
| **age_of_respondent**   | Age of the interviewee                                                                            |
| **gender_of_respondent**| Gender of interviewee: Male or Female                                                            |
| **relationship_with_head** | Interviewee’s relationship with head of household (e.g., Head of Household, Spouse, Child, etc.)|
| **marital_status**      | Marital status (Married/Living together, Divorced/Separated, Widowed, Single/Never Married, Don’t know) |
| **education_level**     | Highest level of education (No formal education, Primary, Secondary, Vocational/Specialised training, Tertiary, Other/Don’t know/RTA) |
| **job_type**            | Type of job (Farming and Fishing, Self employed, Formally employed Government/Private, Informally employed, Remittance Dependent, Government Dependent, Other Income, No Income, Don’t know/Refuse to answer) |

---

## How to Use This Repository

- The repo contains the preprocessing scripts, model training, and Streamlit application code.
- You can use the Streamlit app to input user data and predict the likelihood of having a bank account.
- The model is trained on the demographic and financial data to predict financial inclusion.

---

Feel free to explore and contribute!

