# Create an application to predict your insurance premium cost with Auto AI 

![demo](https://media.github.ibm.com/user/79254/files/37d5e300-8bbf-11ea-9f76-c9dd15d1eef0)

As shown above, this application leverages machine learning models to predict your insurance charges, and how smoking or decreasing your BMI may affect
your insurance charges.

Automation and artificial intelligence (AI) transforms businesses and can help address challenges in areas of healthcare. Many major insurers are exploring how to leverage AI solutions to prevent negative health outcomes before they happen. We will try to predict the insurance premium cost with an intent to create an effective systems which can predict the outcome accurately.

## Description

Using IBMs AutoAI, we automate all the tasks involved in building predictive models for different requirements. We see how AutoAI generates great models quickly which save time and effort and aid in faster decision making process. We create a model that from a data set that includes the age, sex, BMI, number-of-children, smoking preferences, region and charges to predict the health insurance premium cost that an individual will pay.


When the reader has completed this code pattern, they understand how to:

* Quickly set up the services on IBM cloud for model building.
* Ingest the data and initiate the Auto AI process.
* Build different models using Auto AI and evaluate the performance.
* Choose the best model and complete the deployment.
* Generate predictions using the deployed model by making ReST calls.
* Compare the process of using Auto AI and building the model manually.
* Visualize the deployed model using a front end application.


## Architecture Components

![Architecture Components](https://media.github.ibm.com/user/79254/files/f849c100-8b02-11ea-88ed-9f836157f992)

## Flow Description
1. The user creates an IBM Watson Studio Service on IBM Cloud.
2. The user creates an IBM Cloud Object Storage Service and adds that to Watson Studio.
3. The user uploads the insurance charges data file into Watson Studio.
4. The user creates an AutoAI Experiment to predict insurance charges on Watson Studio, and deploys the best performing model.
5. The user uses the Flask web-application to talk to the deployed model and predict an insurane charge.


## Included components
*	[IBM Watson Studio](https://cloud.ibm.com/catalog/services/watson-studio) - IBM Watson® Studio helps data scientists and analysts prepare data and build models at scale across any cloud.
*	[IBM Watson Machine Learning](https://cloud.ibm.com/catalog/services/machine-learning) - IBM Watson® Machine Learning helps data scientists and developers accelerate AI and machine-learning deployment. 
*	[IBM Cloud Object Storage](https://cloud.ibm.com/catalog/services/cloud-object-storage) IBM Cloud™ Object Storage makes it possible to store practically limitless amounts of data, simply and cost effectively.

## Featured technologies
+ [artificial-intelligence](https://developer.ibm.com/technologies/artificial-intelligence/) Build and train models, and create apps, with a trusted AI-infused platform.
+ [Python](https://www.python.org/) Python is an interpreted, high-level, general-purpose programming language.

## Watch the Video - TODO

<!-- [![](docs/ibpVideo.png)](https://www.youtube.com/watch?v=ny8iif6ZXRU) -->

## Prerequisites

This Cloud pattern assumes you have an **IBM Cloud** account. Go to the 
link below to sign up for a no-charge trial account - no credit card required. 
  - [IBM Cloud account](https://tinyurl.com/y4mzxow5)



# Steps
1. [Clone the repo](#step-1-clone-the-repo)
2. [Explore the data (optional)](#step-2-explore-the-data-optional)
3. [Create IBM Cloud services](#step-3-create-ibm-cloud-services)
4. [Create and Run Auto AI experiment](#step-4-create-and-run-auto-ai-experiment)
5. [Create a deployment and test your model](#step-5-create-a-deployment-and-test-your-model)
6. [Create a notebook from your model (optional)](#step-6-create-a-notebook-from-your-model-optional)
7. [Run the application](#step-7-run-the-application)

## Step 1. Clone the repo
Clone this repo onto your computer in the destination of your choice:
```
git clone https://github.ibm.com/Horea-Porutiu/AoT-AutoAI.git
```
This will give you access to the data files in the `data` directory. The data set we 
will use today is an [insurance premiun data set](https://www.kaggle.com/noordeen/insurance-premium-prediction) from Kaggle.




## Step 2. Explore the data (optional)

#### If you want to run the notebook that we will explore below, go to [`notebooks/Claim Amount Exploratory.ipynb`](https://github.ibm.com/Horea-Porutiu/AoT-AutoAI/blob/master/notebooks/Claim%20Amount%20Exploratory.ipynb).
* Within Watson Studio, we explore the data before we create any 
machine learning models. We want to understand our data, and find any trends between 
what we are trying to predict (insurance premium <b>charges</b>) and our features.

* As you can see, once we import our data into a data frame, and call the 
`df_claim.head()` function, we will see the first 5 rows of our data set. 
We can see the features to be `age`, `sex`, `bmi`, `children`, `smoker`,
and `region`.

![scatter](https://media.github.ibm.com/user/79254/files/ed325a80-8a48-11ea-8fcf-d1e9877458ef)

* To check if there is a strong relationship between `bmi` and `charges` we 
create a scatter plot using the seaborn and matplotlib libraries. We 
see that there is no strong correlation between `bmi` and `charges`,
as shown below.

![scatter](https://media.github.ibm.com/user/79254/files/2965bb00-8a49-11ea-81f9-a528fc1e2606)

* To check if there is a strong relationship between `sex` and `charges` we create a box plot. We see that the average claims for males and females are similar, whereas males have a bigger proportion of the higher claims.

![scatter](https://media.github.ibm.com/user/79254/files/32ef2300-8a49-11ea-93aa-990f85eccf9d)

* To check if there is a strong relationship between being a `smoker` and `charges` we create a box plot. We see that if you are a smoker, your claims are much higher on average.

![scatter](https://media.github.ibm.com/user/79254/files/4221a100-8a48-11ea-8104-64f50d8ae92f)

* Let's see if the `smoker` group is well represented. As we see, below, it is. 
There are around 300 smokers, and around 1000 non-smokers.

![scatter](https://media.github.ibm.com/user/79254/files/477eeb80-8a48-11ea-83a0-9a073bf4f176)

* To check if there is a strong relationship between being a `age` and `charges` we create a scatter plot. We see that claim amounts increase with age, and tend to form groups around 12,000, 30,000, and 40,000.

![scatter](https://media.github.ibm.com/user/79254/files/5bc2e880-8a48-11ea-8dad-8effab71a8ac)

If you want to see all of the code, and run the notebook yourself, check the data folder above.

## Step 3. Create IBM Cloud services

First login to your IBM Cloud account. Use the video below for directions on how to create IBM Watson Studio Service.

![watsonStudio](https://media.github.ibm.com/user/79254/files/e493eb80-8626-11ea-87b5-f1c7cf8d50e0)

* After logging into IBM Cloud, click `Proceed` to show that you have read your data rights.

* Click on `IBM Cloud` in the top left corner to ensure you are on the home page.

* Within your IBM Cloud account, click on the top search bar to search for cloud services and offerings. Type in `Watson Studio` and then click on `Watson Studio` under `Catalog Results`.

* This will take you to the Watson Studio service page. There you can name the service as you wish. I named it mine
`Watson-Studio-trial`. You can also choose which data center to create your instance in. The gif above shows mine as 
being created in Dallas.

* For this guide, we will choose the `Lite` service, which is no-charge. This has limited compute; it is enough
to understand the main functionality of the service.

* Once you are satisfied with your service name, and location, and plan, click on create in the bottom-right corner. This creates your Watson Studio instance. 

![createProj](https://media.github.ibm.com/user/79254/files/db5a4d00-862d-11ea-96ce-0872b828932d)

* To launch your Watson Studio service, go back to the home page by clicking on `IBM Cloud` in the top-left corner. There you see your services, and under there you should see your service name. This might take a minute or two to update. 

* Once you see your service that you just created, click on your service name, and this takes you to your 
Watson Studio instance page, which says `Welcome to Watson Studio. Let's get started!`. Click on the `Get Started` button.

* This takes you to the Watson Studio tooling. There you see a heading that says `Start by creating a project` and a button that says `Create Project`. Click on `Create a Project`. Next click on `Create an Empty project`.

* On the create a new project page, name your project. I named mine `insurance-demo`. We also need to associate a IBM Cloud Object store instance, so that we can store our data set.

* Under `Select Storage service` click on the `Add` button. This takes you to the IBM Cloud Object Store service page. Leave the service on the `Lite` tier and then click the `Create` button at the bottom of the page. You are prompted to name the service, and choose the resource group. Once you are have selected a name, click in the resource group on `Confirm`. 

* Once you've confirmed your IBM Cloud Object Store instance, you are taken back to the project page. Click on `refresh` and you should see your newly created Cloud Object Store instance under `Storage`. That's it! Now you can click `Create` at the bottom right of the page to create your first IBM Watson Studio project :) 

![addData](https://media.github.ibm.com/user/79254/files/0e054500-8630-11ea-99dc-7e13ce87bd9d)

* Once you have created your Watson Studio Project, you see a blue `Add to Project` button on the top-right corner of your screen. Click on `Add to Project` and then select `Data`. This brings up a column on the right hand side that says `Data`. 

* In the Data column, click on `browse` to add data from a file. Go into where you cloned your project from 
[Step 1](https://github.ibm.com/Horea-Porutiu/AoT-AutoAI/tree/master#step-1-clone-the-repo) and then navigate
to the `data` folder, and then select `insurance.csv`. 

* Watson Studio takes a couple of seconds to load the data, and then your should see the import has completed. To make sure it has worked properly, you can click on `Assets` on the top of the page, and you should see your 
insurance file under `Data Assets`. 

## Step 4. Create and Run Auto AI experiment

![createAutoAI](https://media.github.ibm.com/user/79254/files/09409100-8630-11ea-804e-ad92728b7f26)

* Once you've created your project, click on the `Add to project` at the top-right of your Watson Studio project page. This  pops up an image with different assets you can choose to add to your project. Click on `Auto AI experiment`.

* This takes your to a page which says `New AutoAI expriment` at the top-left. Name your experiment as you want. I named mine `auto-ai-insurance-demo`.

* Next, we need to add a Watson Machine Learning instance before we can create our Watson AutoAI experiment. On the right side of the screen click on `Associate a Machine Learning instance`. 

* Same as before, select the `Lite` Tier, and click on the `Create` button at the bottom of the page. Name your instance as you wish. I named mine `machine-learning-free`. Choose the location and the resource group and then click on `Confirm` when you are happy with your instance details.

* Once you create your machine learning service, you are taken back to the new AutoAI experiment page. Click on 
`Reload` on the right side of the screen. You should see your newly created machine learning instance. Great job! Click on `Create` on the bottom right part of your screen to create your first AutoAI experiment!

![experimentSettings](https://media.github.ibm.com/user/79254/files/05ad0a00-8630-11ea-94e7-cd47ae3ac941)

* After you create your experiment, you are taken to a page to add a data source to your project. Click on `Select from project` and then add the `insurance.csv` file. Click on `Select asset` to confirm your data source.

* Next, you see that AutoAI processes your data, and you see a `Select your prediction` Column on the 
right. First, let's explore the AutoAI settings to see what you can customize when running your experiment.

* Click on `Experiment settings.` First, you will see the `data source` tab, which will let you omit 
certain columns from your experiment. We have chosen to leave all columns. You can also select the 
training data split. It defaults to 85% training data. The data source tab also shows which metric you  
optimize for. For our regression, it is RMSE (Root Mean Squared Error), and for other types of experiments,
such as Binary Classification, AutoAI defaults to Accuracy. Either way, you can change the metric from this tab depending on your use case.

* Click on the `Prediction` tab from within the `Experiment settings`. There you can select from Binary Classification, Regression, and Multiclass Classificaiton.

* Lastly, you can see the `Runtime` tab from the `Experiment settings` this shows you other experiment details 
you may want to change depending on your use case. 

* Once you are happy with your settings, click on the run `Run Experiment` button on the bottom-right corner of the 
screen.

![compl](https://media.github.ibm.com/user/79254/files/004fbf80-8630-11ea-9c69-e97b12c39bbe)

* Next, your AutoAI experiment runs on its own. You see a progress map on the right side of the screen
which shows which stage of the experiment is running. This may be Hyper Parameter Optimization, feature engineering, 
or some other stage.

* You have different pipelines that are created, and you see the rankings of each model. Each model is ranked based on the metric that you selected. In our case that is the RMSE(Root mean squared error). Given that we want that number to be as small as possible, you can see that in our experiment, the model with the smallest RMSE is at the top of our leaderboard.

* Once the experiment is done, you see `Experiment completed` under the Progress map on the right hand side of
the screen. 

![compl](https://media.github.ibm.com/user/79254/files/38963a00-8a44-11ea-9696-377f268b7af6)

* Now that AutoAI has sucessfully generated eight different models, you can rank the models by different metrics, such as explained variance, root mean squared error, R-Squared, and mean absolute error. Each time you select a different metric, the models will be re-ranked by that metric.

* Let's pick RMSE as our metric. We see that the smallest RMSE value is 4514.389, from Pipeline 8. Click on `Pipeline 8`.

* On the left-hand side, you can see different `Model Evaluation Measures`. For this particular model, you can view the metrics, such as explained variance, RMSE, and other metrics.

* On the left-hand side, you can also see `Feature Transformations`, and `Feature Importance`.

* On the left-hand side, click on `Feature Importance`. You can see here that the most important predictor of our insurance premium is whether you are a `smoker` or `not-smoker`. This is by far the most important feature, with `bmi` coming in as the second most important. This makes sense, given that many companies offer discounts for employees who do not smoke.

## Step 5. Create a deployment and test your model
![compl](https://media.github.ibm.com/user/79254/files/4ea8f800-8a4e-11ea-9da5-f87bff6f4fef)

* Once you are ready to deploy one of the models, click on `Save As` at the top-right corner of the model you want to deploy. Save it as a `Model`. We show you how to save it as a notebook in step 6. 

* Name your model as you want, I named mine `Insurance Premium Predictor - Pattern Demo`.

* Once you have finished saving it as a deployment, you see a green notification at the top right of your screen saying that your model has been sucessfully saved. Click on `View in Project` on that notification at the top-right corner of your screen.

* Next, you are taken to a screen that has the name of the model you just saved. Click on `Deployments` from the Tab in the middle of the screen. 

* Next, click on the `Add Deployment` button on the right-side of the screen. Name your deployment as you want. I named mine `demo-deployment` and then click `Save`.

* On your saved model overview page, you should see your new deployment `demo-deployment` being initialized.

![compl](https://media.github.ibm.com/user/79254/files/caa34000-8a4e-11ea-9142-b1e19a482b94)

* Click on `demo-deployment` or whatever you named your deployment.

* It will take a few minutes for the deployment to be complete. Once it is complete - you see that a `Test` tab appears in the top of the screen. Click on the `Test` tab.

* Here you can test your model. Enter input data such as `age`, `bmi`, `children`, `smoker` and `region`, and then click the `Predict` button at the bottom of the screen.

* As you can see, the model predicted I would get a premium of 4655, when I entered in 
age 27, bmi: 22, children: 0, smoker: no, region: southwest.

* To validate the prediction, I check the data file that we used to train the model, and see
a row that has similar inputs to what I inputted. I found a male, 26 year old, with 0 children,
non-smoker to get a premium of 3,900. This is relatively close to the model's prediction, so 
we know the model is working properly.

## Step 6. Create a notebook from your model (optional)
#### If you want to run the notebook that we will explore below, go to [`notebooks/Insurance Premium Predictor - P8 notebook.ipynb`](https://github.ibm.com/Horea-Porutiu/AoT-AutoAI/blob/master/notebooks/Insurance%20Premium%20Predictor%20-%20P8%20notebook.ipynb).
With AutoAI's latest features, the code that is run to create these models is no more a black box. One or more of these models can be saved as a Jupyter notebook and the python code can be run and enhanced from within. 

### 6.1 Create notebook 
![create notebook](https://media.github.ibm.com/user/9960/files/1c47db00-8ae0-11ea-9066-9bb6137b6ee3)

* Click on `Save As` at the top-right corner of the model, and click `Notebook`. 

* This opens a new tab (be sure to enable pop-up for this website) titled `New Notebook` where in you can edit the default name if you choose to and then click on `Create`. This might take a few minutes to load for the first time. 

![also create notebook](https://media.github.ibm.com/user/9960/files/9a58b180-8ae1-11ea-97ca-f8ec5813f2ed)

* Alternatively, you can also create the notebook from the `Pipeline leaderboard` view (shown above) by clicking on the `Save as` option against the model you want to save followed by selecting `Notebook`. The steps thereon is very similar to the first method discussed above. 

### 6.2 Run notebook
![run notebook](https://media.github.ibm.com/user/9960/files/23e4e800-8e30-11ea-9335-9f4ae2b4e4ba)

* Once the notebook has been created, it is listed under the `Notebooks` section within the `Assets` tab. 
* Clicking on the notebook from the list opens the Jupyter notebook where the code in Python is available. 
* If the notebook is locked, click on the pencil icon on the right tab to be able to run/edit the notebook. 
* Select `Cell` option from the menu list and click `Run All`. This begins runninng all steps in a sequence. Unless an error is encountered, the entire notebook content is executed. 

### 6.3 Analyse notebook content
While understanding the content within the notebook requires prior knowledge of machine learning using python, we encourage you to browse through  [this](https://developer.ibm.com/tutorials/learn-regression-algorithms-using-python-and-scikit-learn/) tutorial to learn the basics of how regression models are built in python. 

In here we will do a highlevel analyses of the notebook that is generated. 

* AutoAI uses [sckikit-learn](https://scikit-learn.org/stable/index.html) for creating machine learning models and for executing the steps in pipelines.  


* [autoai-lib](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/autoai-lib-python.html) is used to transform data while being processed in the pipeline. 


* Following snippet highlights sample code of how auto-ai is used in transforming numerical data and how scikit-learn is used in setting these transformations in a pipeline.
![code snippet-1](https://media.github.ibm.com/user/9960/files/1cbdda00-8e30-11ea-8c2b-ee84388a27f4)


* Here we see the python code that went into setting up Random Forest as the algorithm of choice for regression. 
![code snippet-2](https://media.github.ibm.com/user/9960/files/f8fa9400-8e2f-11ea-8c5b-4f5f5c0875d2)


* Calling the fit method on the pipeline, returns an estimator which is then used to predict a value. The code below shows each of these steps.
![code snippet-3](https://media.github.ibm.com/user/9960/files/1596cc00-8e30-11ea-8905-564f961daae3)


* Finally, the python code that was generated to validate the results and analyse the model performance is seen below. KFold-cross validation techniques have been applied to evaluate the model. The notebook can also be edited to apply other validation techniques and can be re-evaluated.
![code snippet-4](https://media.github.ibm.com/user/9960/files/057eec80-8e30-11ea-8239-a196b1d65cad)


More information on the implementation considerations of AutoAI can be found [here](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/autoai-details.html)

## Step 7. Run the application
The driver code to run the application can be found under the web-app folder within the git repository that was cloned from [Step 1](#step-1-clone-the-repo). To run and test your deployed model through this Python-based user-interface,
you will need to replace the following information within web-app/app.py : 

1) Your Watson Machine Learning (which is associated with this deployed model) `Instance ID` and `apikey`.
1) Your deployed model's deployment URL, so you can make a POST request.
1) Your IBM Cloud IAM token, to authorize yourself. 

Now, we will go into detail on how to gather these credentials. If you already know how to do this, you can
skip the steps below, and  go straight to running the application.


### 7.1 Get Watson Machine Learning Instance ID and apikey

![apikey-instanceID](https://media.github.ibm.com/user/79254/files/4119b680-8e30-11ea-8bc3-97ab1558fc23)

* To get your Watson Machine Learning `Instance ID` and `apikey` first go to `https://cloud.ibm.com/resources` and then 
under `Services` click on the Watson Machine Learning instance that is associated with your Watson Studio
AutoAI experiment.

* Once the Watson Machine Learning service page loads, click on `service credentials` in the left sidebar. 

* From there, expand the `Key Name` by clicking on the down arrow.

* There, you will find your `apikey`, and `Instance ID` keep these handy.

### 7.2 Get model deployment URL

![model-deploy-url](https://media.github.ibm.com/user/79254/files/0f095400-8e32-11ea-89f6-64a89a6f0486)

* From inside Watson Studio, click on your project that you created. 

* From there, click on the `deployments` tab from the top of the screen. Mine is called `Insurance-Premium-Predictor`. 

* Next, click on `Implementation` from the tab at the top of the screen.

* Scroll down to Code Snippets and click on Python.

* Copy the  *deploymnentID* from `/deployments/*******deploymentID*******/predictions` section and paste it into
`web-app/app.py` on line 49 - to complete the POST request URL.

### 7.3 Generate the access token

![model-deploy-url](https://media.github.ibm.com/user/79254/files/07997900-8e39-11ea-82f7-0ee85cc00f90)

* From the command line, type ```curl -V``` to verify if cURL is installed in your system. If cURL is note installed, refer to [this](https://develop.zendesk.com/hc/en-us/articles/360001068567-Installing-and-using-cURL#install) instructions to get it installed.
* Execute the following cURL command to generate your access token, but replace the apikey with the 
apikey we got from [step 7.1](https://github.ibm.com/Horea-Porutiu/AoT-AutoAI#71-get-watson-machine-learning-instance-id-and-apikey) above. 

```
curl -k -X POST \
--header "Content-Type: application/x-www-form-urlencoded" \
--header "Accept: application/json" \
--data-urlencode "grant_type=urn:ibm:params:oauth:grant-type:apikey" \
--data-urlencode "apikey=123456789" \
"https://iam.bluemix.net/identity/token"
```

* Copy and paste the access token into the header in the `web-app/app.py` file. Replace the line
`" TODO: ADD YOUR IAM ACCESS TOKEN FROM IBM CLOUD HERE"` with your token.

![model-deploy-url](https://media.github.ibm.com/user/79254/files/12a0d900-8e3a-11ea-86d8-ccf842fb948e)

* Lastly, input your Watson Machine Learning Instance ID right under where you put your access token.
Replace the line `TODO: ADD YOUR ML INSTANCE ID HERE ` with your instance ID from [step 7.1](https://github.ibm.com/Horea-Porutiu/AoT-AutoAI#71-get-watson-machine-learning-instance-id-and-apikey) above.

* Great job! You are ready to run the application! 

### 7.4 Install dependencies, and run the app

Note, this app is tested on this version of Python 3.8.2

Within the `web-app` directory, run the following command: 

```
pip3 install flask flask-wtf urllib3 requests
```

![run-app](https://media.github.ibm.com/user/79254/files/4a5c5080-8e3b-11ea-8dde-1391459dcc0d)

Next, run the following command to start the flask application.

```
flask run
```

* Go to `127.0.0.1:5000` in your browser to view the application. Go ahead and fill in the form, and click on the `Predict`
button to see your predicted charges based on your data. 

* As is expected, if you are a smoker, this will drastically increase your insurance charges. 

## Bonus Section - Visualize the data and share your findings via Cognos Dashboard Embedded.
* You can add a Dashboard which is a lean version of Cognos Dashboard available on IBM cloud from "Add to Project" option in your watson Studio project.

* You can start finding patterns in your data by easily visualizing various data points. This can get your exploration started within few minutes and with no coding involved
![Cognos-1](https://media.github.ibm.com/user/34798/files/46c6ca80-8e37-11ea-9974-d76d2cc2db87)

* From visualizing this data you can see the relation in the data points, how Gender, BMI, # of children  and smoking might influence the insurance premium.

* Dashboards are very interactive and makes it easy to play with data.
![Cognos-2](https://media.github.ibm.com/user/34798/files/81c8fe00-8e37-11ea-91a0-ef94321258bf)

* You can also pivot and summarize your measures to quickly look at all your measures
![Cognos-3](https://media.github.ibm.com/user/34798/files/855c8500-8e37-11ea-85ca-3406995f8e2d)

* Stop working in Silos and share your findings with your team in two clicks.
![Cognos-3](https://media.github.ibm.com/user/34798/files/8ab9cf80-8e37-11ea-9554-a85f6ad6186f)

## Related Links
* [Fraud Prediction Using Auto AI](https://github.com/IBM/predict-fraud-using-auto-ai)
* [Use AutoAI to predict Customer Churn tutorial](https://developer.ibm.com/tutorials/watson-studio-auto-ai/)
* [Predict Loan Default with AutoAI tutorial](https://developer.ibm.com/tutorials/generate-machine-learning-model-pipelines-to-choose-the-best-model-for-your-problem-autoai/)

## License
This code pattern is licensed under the Apache Software License, Version 2. Separate third-party code objects invoked within this code pattern are licensed by their respective providers pursuant to their own separate licenses. Contributions are subject to the [Developer Certificate of Origin, Version 1.1 (DCO)](https://developercertificate.org/) and the [Apache Software License, Version 2](https://www.apache.org/licenses/LICENSE-2.0.txt).

[Apache Software License (ASL) FAQ](https://www.apache.org/foundation/license-faq.html#WhatDoesItMEAN)

