from  flask import Flask,request,jsonify,render_template
import numpy as np
import pickle
import sklearn

app = Flask(__name__)
model = pickle.load(open("crop2.pkl",'rb'))

@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict" ,methods=['POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    res = str(prediction)[1:-1]

    return render_template("index.html",prediction_text="{}".format(res))




if __name__ == "__main__":
    app.run(debug=True)