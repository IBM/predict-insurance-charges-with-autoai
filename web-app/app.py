from flask import Flask, url_for, render_template, redirect
from forms import PredictForm
from flask import request, sessions
import requests
from flask import json
from flask import jsonify
from flask import Request
from flask import Response
import urllib3
import json
from flask_wtf import FlaskForm

app = Flask(__name__, instance_relative_config=False)
app.secret_key = 'development key' #you will need a secret key

#app.config.from_object('config.Config')


if __name__ == "__main__":
  app.run()

@app.route('/', methods=('GET', 'POST'))
def predict():
    print("I am here")
    form = PredictForm()
    if form.submit():
        print("inside if statement")
        # NOTE: generate iam_token and retrieve ml_instance_id based on provided documentation
        header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + "PUT YOUR IAM Token here",
                  'ML-Instance-ID': "PUT ML INSTANCE ID"}
        python_object = [form.age.data, form.sex.data, str(form.bmi.data), form.children.data, form.smoker.data, form.region.data]
        #Transform python objects to  Json
        json_object = json.dumps(python_object)
        # NOTE: manually define and pass the array(s) of values to be scored in the next line
        payload_scoring = {"input_data": [{"fields": ["age", "sex", "bmi", "children", "smoker", "region"], "values": json_object }]}

        print("After payload statement")
        response_scoring = requests.post("https://us-south.ml.cloud.ibm.com/v4/deployments/1ce09d1c-d84d-4b20-9069-f2a78f5a9b3a/predictions", json=payload_scoring, headers=header)
        print("After response scoring  statement")

        print("Scoring response")
        print(json.loads(response_scoring.text))
        ab = json.loads(response_scoring.text)
        form.abc= ab # this returns the response back to the front page

    return render_template('index.html', form=form)
