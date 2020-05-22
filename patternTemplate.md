# Short title

Create a machine learning web-app to predict your insurance premium cost  

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

# Videos

Demo 1/3: Data exploration and visualization
> https://www.youtube.com/watch?v=9JuiqVXvQ74

Demo 2/3: Running AutoAI
> https://www.youtube.com/watch?v=ilw6O5HwtY0

Demo 3/3: Connecting model API to a web-app
> https://www.youtube.com/watch?v=sOtezE-YNPU

# Summary

As we see the value of gross insurance premiums worldwide continue to skyrocket past 5 trillion dollars,
we know that most of these costs are preventable. For example, just by eliminating smoking, and lowering
your BMI by a few points could mean shaving thousands of dollars off of your premium charges. In this 
application, we study the effects of age, smoking, BMI, gender, and region to determine how much of 
a difference these factors can make on your insurance premium.  By using our 
application, customers see the radical difference their lifestyle choices make on their insurance 
charges. By leveraging AI and machine learning, we help customers understand just how much smoking increases their premium, by predicting how much they will have to pay within seconds.

# Technologies

+ [artificial-intelligence](https://developer.ibm.com/technologies/artificial-intelligence/) Build and train models, and create apps, with a trusted AI-infused platform.
+ [Python](https://www.python.org/) Python is an interpreted, high-level, general-purpose programming language.


# Description

Using IBM AutoAI, we automate all the tasks involved in building predictive models for different requirements. You see how AutoAI generates great models quickly which save time and effort and aid in faster decision-making process. You create a model that from a data set that includes the age, sex, BMI, number-of-children, smoking preferences, region and charges to predict the health insurance premium cost that an individual pays.

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
5. AutoAI uses Watson Machine Learning to create several models, and the user deploys the best performing model.
6. The user uses the Flask web-application to connect to the deployed model and predict an insurance charge.

# Components and services
*	[IBM Watson Studio](https://console.bluemix.net/docs/services/blockchain/howto/ibp-v2-deploy-iks.html#ibp-v2-deploy-iks) gives you total control of your blockchain network with a user interface that can simplify and accelerate your journey to deploy and manage blockchain components on the IBM Cloud Kubernetes Service.
*	[IBM Watson Machine Learning](https://cloud.ibm.com/catalog/services/machine-learning) - IBM Watson® Machine Learning helps data scientists and developers accelerate AI and machine-learning deployment. 
*	[IBM Cloud Object Storage](https://cloud.ibm.com/catalog/services/cloud-object-storage) - IBM Cloud™ Object Storage makes it possible to store practically limitless amounts of data, simply and cost effectively.

# Runtimes

* Python 3.8.2

## Related IBM Developer Content
* [Fraud Prediction Using AutoAI](https://github.com/IBM/predict-fraud-using-auto-ai)
* [Use AutoAI to predict Customer Churn tutorial](https://developer.ibm.com/tutorials/watson-studio-auto-ai/)
* [Predict Loan Default with AutoAI tutorial](https://developer.ibm.com/tutorials/generate-machine-learning-model-pipelines-to-choose-the-best-model-for-your-problem-autoai/)

## Related links
* [What is AutoAI with IBM Watson Studio](https://www.ibm.com/cloud/watson-studio/autoai)
* [What is linear regression](https://www.statisticssolutions.com/what-is-linear-regression/)
