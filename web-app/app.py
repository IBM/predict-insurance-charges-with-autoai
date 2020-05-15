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
# from flask_wtf import FlaskForm

app = Flask(__name__, instance_relative_config=False)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = 'development key' #you will need a secret key

if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0')

@app.route('/', methods=('GET', 'POST'))

def startApp():
    form = PredictForm()
    return render_template('index.html', form=form)

@app.route('/predict', methods=('GET', 'POST'))
def predict():
    form = PredictForm()
    if form.submit():

        # NOTE: generate iam_token and retrieve ml_instance_id based on provided documentation
        header = {'Content-Type': 'application/json', 'Authorization': 'Bearer '
                 + "eyJraWQiOiIyMDIwMDQyNTE4MjkiLCJhbGciOiJSUzI1NiJ9.eyJpYW1faWQiOiJpYW0tU2VydmljZUlkLTc1MWI5YjE4LTI3YzgtNDFmMC1iMGZhLTk4MGJiYjc4ZDBjZSIsImlkIjoiaWFtLVNlcnZpY2VJZC03NTFiOWIxOC0yN2M4LTQxZjAtYjBmYS05ODBiYmI3OGQwY2UiLCJyZWFsbWlkIjoiaWFtIiwiaWRlbnRpZmllciI6IlNlcnZpY2VJZC03NTFiOWIxOC0yN2M4LTQxZjAtYjBmYS05ODBiYmI3OGQwY2UiLCJuYW1lIjoiU2VydmljZSBjcmVkZW50aWFscy0xIiwic3ViIjoiU2VydmljZUlkLTc1MWI5YjE4LTI3YzgtNDFmMC1iMGZhLTk4MGJiYjc4ZDBjZSIsInN1Yl90eXBlIjoiU2VydmljZUlkIiwiYWNjb3VudCI6eyJ2YWxpZCI6dHJ1ZSwiYnNzIjoiZDVhNDRjMTg3ZmFmYjJkZWRhMWE4NTJjZWYzMTEzNmMifSwiaWF0IjoxNTg5NTA4ODQzLCJleHAiOjE1ODk1MTI0NDMsImlzcyI6Imh0dHBzOi8vaWFtLmJsdWVtaXgubmV0L2lkZW50aXR5IiwiZ3JhbnRfdHlwZSI6InVybjppYm06cGFyYW1zOm9hdXRoOmdyYW50LXR5cGU6YXBpa2V5Iiwic2NvcGUiOiJpYm0gb3BlbmlkIiwiY2xpZW50X2lkIjoiZGVmYXVsdCIsImFjciI6MSwiYW1yIjpbInB3ZCJdfQ.AE5CQ-8J43uP0rrG2QiBMWBnGn_IGzDdPZmVhd4hbDURXd1AGGgP-wpwKu329mSiQ2zlHL_8OUkChwrLg5jRVB4GDbD-18ECyM0AwouZII4dAsCiKdj0CduZjzj5nS2nzTnmUgMFwYNLPmfZDas1Mz3EbD6Osi8O0-IjHAkt-Y1ahGCeV5lWI-dtHnS7sTFt-j1oUxcQf6rNs87QdGgz1fBTE3QcNnvZpNgCCq6VOYXJWVzuAYAapdoYwKxybEXlNe4YFBYk4xkNi8pefPf99taG4Tp2ojpxjMXcEZ9wdgY9j_2upreUrg44zYyj_uGXx1ubNfUnOWSPrFTozsRagg","refresh_token":"OKA-WGoewazY90sIbn4OGDIfujYeGo-c-TpaVlo1uUViQqVRnpEOHrXUL3Ug-BvmR9QzUAmfmjRl9Nw9Vvjf9s8gFbIjplxld1EbOrLkKGgnm_IbPKXfZS6xMB674AvoA-wMJzdfmKPHyKNFjzI_eT8ytLrw6FzoRucLAcHnHYh_YAJsAG5jZHYsdFl2c0lvRFHRwnqKd3svEbxZhSpOoj9Qx_JcohOd2DwmJFgr3H2x2Zsc0RfIlx5qG6YpRgqPJlMJy5iF6Di2J-zUfu0OrQ5WRsA_c_sXYiXA8l3fWA_BI21c8JJfydZpHOQCyuJ9ENOXeDpCZwedi2JuDw4F1ZW3htioumcwbhAkubpQNuJzUFP_O9MFD6AfLMCs9SrC-kd9Ti0GfP224GndCaoTnfXWov4CGgqLVEfdxe2Biiyl_qFrnoI47mcet0QBSXEgq0p-GRCUxAmRLHAndR0KizIRfCA1-1TI5pWAef_bQ3fvKb8xY3_QAO44uX5FHcTPFzUwRJIBO13ayspbdVH-emIw1g3kt6vE2eGiwznES6eJEMHvUARqEUYwxUd30DeZD9LSxUpJxCycW6gj4jQMcT0319Zu8lRVmI7_LTVp0EwP0MBRVH_Gv8kcmpZ7D2L4qjkmr2sjBSeztbmy6mWduSB1oRfY2N4vZYoTJyLjaY7ZxOvsxYopHEfzj2YU80s6g_TSCO4w5XZx0wL5Lfi90QtiPcW6e7sowiu_ECZCAD2odCx1wbDizBtMTJ8z0R-7jHZqqxUhmMZ1rLsE6-WcaQ8CHyzmWgjH9kEEwUHzzzUpW2UQW461xX-Bi_2rBlvtEG6IVtkTDIyPIw5k8giGmgguc_tVcGz3NVRvrwzbA0-pYJWlj2WYHRWdUbKYY1acgK-agQX-jz8V0x5GTu639wejpKea9G711X26N38X-1Fhe2MMbUUa1UO0ySNFTtJlHHRTza9xMJivc00tsn8H2AP-eqtfJThrpZvgwhaLR6Ms2gk3iXztcjrsIWZxkQ8ZcMbOGBVdrt79MZjuhOAI0nWiF2VtZ_uE9UU2C9T0LSDvYA",
                  'ML-Instance-ID': "37cd6426-ce10-4c03-90cf-466b4b630f35"}

        if(form.bmi.data == None): 
          python_object = []
        else:
          python_object = [form.age.data, form.sex.data, float(form.bmi.data),
            form.children.data, form.smoker.data, form.region.data]
        #Transform python objects to  Json

        userInput = []
        userInput.append(python_object)

        # NOTE: manually define and pass the array(s) of values to be scored in the next line
        payload_scoring = {"input_data": [{"fields": ["age", "sex", "bmi",
          "children", "smoker", "region"], "values": userInput }]}

        response_scoring = requests.post("https://us-south.ml.cloud.ibm.com/v4/deployments/a23b3011-a0c7-429d-85cd-63aac387e672/predictions", json=payload_scoring, headers=header)

        output = json.loads(response_scoring.text)
        print(output)
        for key in output:
          ab = output[key]
        

        for key in ab[0]:
          bc = ab[0][key]
        
        roundedCharge = round(bc[0][0],2)

  
        form.abc = roundedCharge # this returns the response back to the front page
        return render_template('index.html', form=form)

