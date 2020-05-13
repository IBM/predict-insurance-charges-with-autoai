# Short title

> Create a machine learning web-app to predict your insurance premium cost  

# Long title

Create a web-application that uses linear regression to predict your insurance premium cost with IBM Watson Studio and Auto AI

# Author

* Horea Porutiu - horea.porutiu@ibm.com
* Samaya Madhavan - smadhava@us.ibm.com
* Irina Saburova - irina.saburova@us.ibm.com
* Paul Bastide - pbastide@us.ibm.com
* Anjini Kumar - gaurak@cn.ibm.com
* Maria Rita Villari - mvillari@us.ibm.com
* Osai Osaigbovo - ooosaigb@us.ibm.com
* Venita Glasfurd - vhglasfu@us.ibm.com


# URLs

### Github repo

> https://github.com/IBM/predict-insurance-charges-with-ai

# Summary

The U.S. insurance industry net pr to U.S. industry premiums totaling 1.22 trillion 
in 2018. N 
As shown above, this application leverages machine learning models to predict your insurance charges, and how smoking or decreasing your BMI affects
insurance premiums.

Automation and artificial intelligence (AI) transforms businesses and helps address challenges in areas of healthcare. Many insurers are exploring how to leverage AI solutions to improve health outcomes for patients under their care. You predict an insurance premium cost with an intent to create an effective solution to accurately predict the insurance premium outcomes.

# Technologies

+ [artificial-intelligence](https://developer.ibm.com/technologies/artificial-intelligence/) Build and train models, and create apps, with a trusted AI-infused platform.
+ [Python](https://www.python.org/) Python is an interpreted, high-level, general-purpose programming language.


# Description

Using IBM AutoAI, you automate all the tasks involved in building predictive models for different requirements. You see how AutoAI generates great models quickly which save time and effort and aid in faster decision-making process. You create a model that from a data set that includes the age, sex, BMI, number-of-children, smoking preferences, region and charges to predict the health insurance premium cost that an individual pays.

When you have completed this code pattern, you understand how to:

* Setup, quickly, the services on IBM Cloud for building the model.
* Ingest the data and initiate the AutoAI process.
* Build different models using AutoAI and evaluate the performance.
* Choose the best model and complete the deployment.
* Generate predictions using the deployed model by making REST calls.
* Compare the process of using AutoAI and building the model manually.
* Visualize the deployed model using a front-end application.

### Architecture Components

![Architecture Components](https://media.github.ibm.com/user/21063/files/3b77e580-913c-11ea-9dea-425b1d4f4ee0)

## Flow Description
1. The user creates an IBM Watson Studio Service on IBM Cloud.
2. The user creates an IBM Cloud Object Storage Service and adds that to Watson Studio.
3. The user uploads the insurance premium data file into Watson Studio.
4. The user creates an AutoAI Experiment to predict insurance premium on Watson Studio
5. AutoAI uses Watson Machine Learning to create serveral models, and the user deploys the best performing model.
6. The user uses the Flask web-application to connect to the deployed model and predict an insurance charge.

# Components and services
*	[IBM Watson Studio](https://console.bluemix.net/docs/services/blockchain/howto/ibp-v2-deploy-iks.html#ibp-v2-deploy-iks) gives you total control of your blockchain network with a user interface that can simplify and accelerate your journey to deploy and manage blockchain components on the IBM Cloud Kubernetes Service.

# Runtimes

* Python

## Related IBM Developer Content
* [Fraud Prediction Using AutoAI](https://github.com/IBM/predict-fraud-using-auto-ai)
* [Use AutoAI to predict Customer Churn tutorial](https://developer.ibm.com/tutorials/watson-studio-auto-ai/)
* [Predict Loan Default with AutoAI tutorial](https://developer.ibm.com/tutorials/generate-machine-learning-model-pipelines-to-choose-the-best-model-for-your-problem-autoai/)


## Related Links
* [Fraud Prediction Using AutoAI](https://github.com/IBM/predict-fraud-using-auto-ai)
* [Use AutoAI to predict Customer Churn tutorial](https://developer.ibm.com/tutorials/watson-studio-auto-ai/)
* [Predict Loan Default with AutoAI tutorial](https://developer.ibm.com/tutorials/generate-machine-learning-model-pipelines-to-choose-the-best-model-for-your-problem-autoai/)

