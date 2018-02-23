import os
import pickle
import pandas as pd
esc
from flask import Flask, render_template, request


app = Flask(__name__)

# initial page for FindMyJob
@app.route('/')
def index():
    return render_template("index.html")

# send user input with POST method
@app.route('/desc', methods=['POST'])
def desc():
    # todo: user input from form tag with POST method
    # You should solve this method
    # return response within result()
    # like this `return result(responses)`
    
    # load data
    df = pd.read_csv("./data/filtered_labeled_train_40.csv")
    job_names = df.job_title.astype('category').unique()

    # user input string
    user_input = [request.form['description']]

    # load model
    clf = 'model-v1.pk'
    print("Loading the models...")
    with open('./models/'+clf, 'rb') as f:
        loaded_model = pickle.load(f)

    # prediction
    prob = loaded_model.predict_proba(user_input)[0]
    df_prob = pd.DataFrame(data=[job_names, prob], columns=["job_name", "prob"])
    sorted_df = df_prob.sort_values(by=['prob'], ascending=False)
    
    responses = sorted_df.to_json(orient="records")

    return result(responses)

@app.route('/result')
def result(responses):
    return render_template("result.html", responses=responses)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080)
