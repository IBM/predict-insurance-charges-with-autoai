## Running `Claim Amount Exploratory.ipynb`

You will need to either do a local import of the `insurance.csv` file. I.e. have the file on your computer and import it from there. 

Or you could upload the file into Watson Studio (into an instance of IBM Cloud Object Storage) and then click on `Import to notebook`
and pick the `insurance.csv` file. After that, running the 
first cell should work successfully.

So - within Watson Studio, you can click on `Add to project` then `Data`. 

Next, the sidebar on the right will open up. and then click on `browse` and add
the insurance data set we downloaded from [Kaggle](https://www.kaggle.com/noordeen/insurance-premium-prediction). 

![addData](https://user-images.githubusercontent.com/10428517/87363039-2301be00-c525-11ea-9aab-6287ccdd48be.png)

Next, open the Claim Exploratory Analysis notebook in Watson Studio, and then
click on the pencil icon in the top bar to edit your notebook. You should see 
that your runtime is being instantiated.

![Screen Shot 2020-07-13 at 4 26 24 PM](https://user-images.githubusercontent.com/10428517/87363262-a58a7d80-c525-11ea-8d6f-2bec6eba45e4.png)

You can comment out the first cell. From the side bar, click on the
icon with `0001`in the top-right corner. There, you will see your files. 
Then click on `insurance.csv` and `Insert to code`. 

![Screen Shot 2020-07-13 at 4 28 32 PM](https://user-images.githubusercontent.com/10428517/87363894-08c8df80-c527-11ea-804b-2983fbc505a4.png)

Then you should see
something like this when you run the code. 

![Screen Shot 2020-07-13 at 4 28 32 PM](https://user-images.githubusercontent.com/10428517/87363787-c30c1700-c526-11ea-9dcc-1ae16454b8fc.png)

Nice! Now you are ready to run the notebook and run the visualizations. 