# Create an application to predict your insurance premium cost with Auto AI 

### Summary

Automation and artificial intelligence (AI) transforms businesses and can help address challenges in areas of healthcare. Many major insurers are exploring how to leverage AI solutions to prevent negative health outcomes before they happen. We will try to predict the insurance premium cost with an intent to create an effective systems which can predict the outcome accurately.

### Description

Using IBMs Auto AI, we will automate all the tasks involved in building predictive models for different requirements. We will get to see how Auto AI can churn out great models quickly which will save time and effort and aid in faster decision making process. We will create a model that from a data set that includes the age, sex, BMI, number of children, smoking preferences,  region and charges to predict the health insurance premium cost that an individual will pay.


When the reader has completed this code pattern, they will understand how to :

* Quickly set up the services on IBM cloud for model building.
* Ingest the data and initiate the Auto AI process.
* Build different models using Auto AI and evaluate the performance.
* Choose the best model and complete the deployment.
* Generate predictions using the deployed model by making ReST calls.
* Compare the process of using Auto AI and building the model manually.
* Visualize the deployed model using a front end application.


### Example Flow Diagram - TODO (Need to change)
<br>
<p align="center">
  <img src="docs/app-architecture.png">
</p>
<br>

### TODO - Flow Description
1. The user creates an Auto AI Project within IBM Watson Studio


## Included components
*	[IBM Watson Studio](https://console.bluemix.net/docs/services/blockchain/howto/ibp-v2-deploy-iks.html#ibp-v2-deploy-iks) 
*	[IBM Watson Machine Learning](https://console.bluemix.net/docs/services/blockchain/howto/ibp-v2-deploy-iks.html#ibp-v2-deploy-iks)
*	[IBM Cloud Object Storage](https://console.bluemix.net/docs/services/blockchain/howto/ibp-v2-deploy-iks.html#ibp-v2-deploy-iks) store smt

## Featured technologies
+ [artificial-intelligence](https://developer.ibm.com/technologies/artificial-intelligence/) Build and train models, and create apps, with a trusted AI-infused platform.
+ [Python](https://www.python.org/) Python is an interpreted, high-level, general-purpose programming language.

## Watch the Video - TODO

<!-- [![](docs/ibpVideo.png)](https://www.youtube.com/watch?v=ny8iif6ZXRU) -->

## Prerequisites

This Cloud pattern assumes you have an **IBM Cloud account**. Go to the 
link below to sign up for a free trial account - no credit card required. 
  - [IBM Cloud account](https://tinyurl.com/y4mzxow5)


# Steps TODO
1. [Clone the repo](#step-1-clone-the-repo)
2. [Explore the data](#step-2-explore-the-data)
3. [Create IBM Cloud services](#step-3-create-ibm-cloud-services)
4. [Create and Run Auto AI experiment](#step-4-create-and-run-auto-ai-experiment)
5. [Create a notebook from your model](#step-5-create-a-notebook-from-your-model)
6. [Run the application](#step-6-run-the-application)

## Step 1. Clone the repo

Clone this repo onto your computer in the destination of your choice:
```
git clone https://github.ibm.com/Horea-Porutiu/AoT-AutoAI.git
```
This will give you access to the data files in the `data` directory. These data sets 
are from Kaggle and `data.boston.gov`.

## Step 2. Explore the data
* Within Watson Studio, we will first do some data exploration before we create any 
machine learning models. We want to understand our data, and find any trends between 
what we are trying to predict (insurance premium <b>charges</b>) and our features.

* As you can see, once we import our data into a data frame, and call the 
`df_claim.head()` function, we will see the first 5 rows of our data set. 
We can see the features to be `age`, `sex`, `bmi`, `children`, `smoker`,
and `region`.

![scatter](https://media.github.ibm.com/user/79254/files/ed325a80-8a48-11ea-8fcf-d1e9877458ef)

* To check if there is a strong relationship between `bmi` and `charges` we 
can create a scatter plot using the seaborn and matplotlib libraries. We 
can see that there is no strong correlation between `bmi` and `charges`,
as shown below.

![scatter](https://media.github.ibm.com/user/79254/files/2965bb00-8a49-11ea-81f9-a528fc1e2606)

* To check if there is a strong relationship between `sex` and `charges` we 
can create a box plot. We 
can see that the average claims for males and females are similar, but males have a
bigger proportion of the higher claims.

![scatter](https://media.github.ibm.com/user/79254/files/32ef2300-8a49-11ea-93aa-990f85eccf9d)

* To check if there is a strong relationship between being a `smoker` and `charges` we 
can create a box plot. We 
can see that if you are a smoker, your claims are much higher on average.

![scatter](https://media.github.ibm.com/user/79254/files/4221a100-8a48-11ea-8104-64f50d8ae92f)

* Let's see if the `smoker` group is well represented. As we can see below, it is. 
There are around 300 smokers, and around 1000 non-smokers.

![scatter](https://media.github.ibm.com/user/79254/files/5bc2e880-8a48-11ea-8dad-8effab71a8ac)


## Step 3. Create IBM Cloud services

First login to your IBM Cloud account. Use the video below for directions on how 
to create IBM Watson Studio Service.

![watsonStudio](https://media.github.ibm.com/user/79254/files/e493eb80-8626-11ea-87b5-f1c7cf8d50e0)
* After logging into IBM Cloud, click `Proceed` to show that you have read your data rights.

* Click on `IBM Cloud` in the top left corner to ensure you are on the home page.

* Within your IBM Cloud account, click on the top search bar to search for cloud services and offerings. Type in `Watson Studio` and then click on `Watson Studio` under `Catalog Results`.

* This will take you to the Watson Studio service page. There you can name the service as you wish. I named it mine
`Watson-Studio-freetrial`. You can also choose which data center to create your instance in. The gif above shows mine as 
being created in Dallas.

* For this guide, we will choose the `Lite` service, which is free. This has limited compute, but will be enough
to understand the main functionality of the service.

* Once you are satisfied with your service name, and location, and plan, click on create in the bottom-right corner. This will create your Watson Studio instance. 

![createProj](https://media.github.ibm.com/user/79254/files/db5a4d00-862d-11ea-96ce-0872b828932d)

* To launch your Watson Studio service, go back to the home page by clicking on `IBM Cloud` in the top-left corner. There you will see your services, and under there you should see your service name. This might take a minute or two 
to update. 

* Once you see your service that you just created, click on your service name, and this will take you to your 
Watson Studio instance page, which will say `Welcome to Watson Studio. Let's get started!`. Click on the `Get Started` button.

* This will take you to the Watson Studio tooling. There you will see a heading that says `Start by creating a project` and a button that says `Create Project`. Click on `Create a Project`. Next click on `Create an Empty project`.

* On the create a new project page, name your project. I named mine `insurance-demo`. We also need to associate a IBM Cloud Object store instance, so that we can store our data set.

* Under `Select Storage service` click on the `Add` button. This will take you to the IBM Cloud Object Store service page. Leave the service on the `Lite` tier and then click the `Create` button at the bottom of the page. You will be prompted to name the service, and choose the resource group. Once you are happy with the naming, and the resource group on `Confirm`. 

* Once you've confirmed your IBM Cloud Object Store instance, you will be taken back to the project page. Click on `refresh` and you should see your newly created Cloud Object Store instance under `Storage`. That's it! Now you can click `Create` at the bottom right of the page to create your first IBM Watson Studio project :) 

![addData](https://media.github.ibm.com/user/79254/files/0e054500-8630-11ea-99dc-7e13ce87bd9d)

* Once you have created your Watson Studio Project, you should see a blue `Add to Project` button on the top-right
corner of your screen. Click on `Add to Project` and then select `Data`. This will bring up a column on the right 
hand side that says `Data`. 

* In the Data column, click on `browse` to add data from a file. Go into where you cloned your project from 
[Step 1](https://github.ibm.com/Horea-Porutiu/AoT-AutoAI/tree/master#step-1-clone-the-repo) and then navigate
to the `data` folder, and then select `insurance.csv`. 

* Watson Studio will take a couple of seconds to load the data, and then your should see the import has completed. To make sure it has worked properly, you can click on `Assets` on the top of the page, and you should see your 
insurance file under `Data Assets`. 


## Step 4. Create and Run Auto AI experiment

![createAutoAI](https://media.github.ibm.com/user/79254/files/09409100-8630-11ea-804e-ad92728b7f26)

* Once you've created your project, click on the `Add to project` at the top-right of your Watson Studio project page. This will pop up an image with different assets you can choose to add to your project. Click on `Auto AI experiment`.

* This will take your to a page which says `New AutoAI expriment` at the top-left. Name your experiment as you want. I named mine `auto-ai-insurance-demo`.

* Next, we need to add a Watson Machine Learning instance before we can create our Watson AutoAI experiment. On the right side of the screen click on
`Associate a Machine Learning instance`. 

* Same as before, select the `Lite` Tier, and click on the `Create` button at the bottom of the page. Name your instance as you wish. I named mine `machine-learning-free`. Choose the location and the resource group and then click on `Confirm` when you are happy with your instance details.

* Once you create your machine learning service, you will be taken back to the new AutoAI experiment page. Click on 
`Reload` on the right side of the screen. You should see your newly created machine learning instance. Great job! Click on `Create` on the bottom right part of your screen to create your first AutoAI experiment!

![experimentSettings](https://media.github.ibm.com/user/79254/files/05ad0a00-8630-11ea-94e7-cd47ae3ac941)

* After you create your experiment, you will be taken to a page to add a data source to your project. Click on `Select from project` and then add the `insurance.csv` file. Click on `Select asset` to confirm your data source.

* Next, you will see that AutoAI processes your data, and you will see a `Select your prediction` Column on the 
right. First, let's explore the AutoAI settings to see what you can customize when running your experiment.

* Click on `Experiment settings.` First, you will see the `data source` tab, which will let you omit 
certain columns from your experiment. We have chosen to leave all columns. You can also select the 
training data split. It defaults to 85% training data. The data source tab also shows which metric you will 
optimize for. For our regression, it will be RMSE (Root Mean Squared Error) but for other types of experiments,
such as Binary Classification, AutoAI will default to accuracy. Either way, you can change the metric from this tab depending on your use case.

* Click on the `Prediction` tab from within the `Experiment settings`. There you can select from Binary Classification, Regression, and Multiclass Classificaiton.

* Lastly, you can see the `Runtime` tab from the `Experiment settings` this will show you other experiment details 
you may want to change depending on your use case. 

* Once you are happy with your settings, click on the run `Run Experiment` button on the bottom-right corner of the 
screen.

![compl](https://media.github.ibm.com/user/79254/files/004fbf80-8630-11ea-9c69-e97b12c39bbe)

* Next, your AutoAI experiment will run on its own. You will see a progress map on the right side of the screen
which shows which stage of the experiment is running. This may be Hyperparameter optimization, feature engineering, 
or some other stage.

* You will have different pipelines that will be created, and you will see the rankings of each model. Each model 
will be ranked based on the metric that you selected. In our case that is the RMSE(Root mean squared error). Given 
that we want that number to be as small as possible, you can see that in our experiment, the model with the smallest RMSE is at the top of our leaderboard.

* Once the experiment is done, you will see `Experiment completed` under the Progress map on the right hand side of
the screen. 

![compl](https://media.github.ibm.com/user/79254/files/38963a00-8a44-11ea-9696-377f268b7af6)

* Now that AutoAI has sucessfully generated eight different models, you can rank the models by 
different metrics, such as explained variance, root mean squared error, R-Squared, and mean 
absolute error. Each time you select a different metric, the models will be re-ranked by
that metric.

* Let's pick RMSE as our metric. We see that the smallest RMSE value is 4514.389, from 
Pipeline 8. Click on `Pipeline 8`.

* On the left-hand side, you can see different `Model Evaluation Measures`. For this particular
model, you can view the metrics, such as explained variance, RMSE, and other metrics.

* On the left-hand side, you can also see `Feature Transformations`, and `Feature Importance`.

* On the left-hand side, click on `Feature Importance`. You can see here that the most 
important predictor of our insurance premium is whether you are a smoker or not. This is by
far the most important feature, with bmi coming in as the second most important. This makes 
sense, given that many companies offer discounts for employees who do not smoke.

## Step 5. Create a notebook from your model
## Step 6. Run the application




## Related Links TODO
<!-- * [Hyperledger Fabric Docs](http://hyperledger-fabric.readthedocs.io/en/latest/)
* [IBM Code Patterns for Blockchain](https://developer.ibm.com/patterns/category/blockchain/) -->

## License
This code pattern is licensed under the Apache Software License, Version 2. Separate third-party code objects invoked within this code pattern are licensed by their respective providers pursuant to their own separate licenses. Contributions are subject to the [Developer Certificate of Origin, Version 1.1 (DCO)](https://developercertificate.org/) and the [Apache Software License, Version 2](https://www.apache.org/licenses/LICENSE-2.0.txt).

[Apache Software License (ASL) FAQ](https://www.apache.org/foundation/license-faq.html#WhatDoesItMEAN)

