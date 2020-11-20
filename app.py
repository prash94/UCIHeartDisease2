import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create a flask app.
app = Flask(__name__)

# Load the pickle file.
filename = 'model1.pickle'
model = pickle.load(open(filename,'rb'))


# This renders the template called index.html, which is our home page.
@app.route('/')
def home():
    return render_template('index.html')


# Create a POST method to capture feature inputs from the template using the request library.
# Feature inputs are then passed onto the predict method.
# /predict is being called by the route code below
@app.route('/predict', methods=['POST'])
def predict():
    # for rendering results in HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]

    # prediction_text is returned here will replace {{ prediction_text }} on the template
    return render_template('index.html',
                           prediction_text='Risk of this patient to have a heart disease is {}'.format(output))


# main function to run this flask app
if __name__ == "__main__":
    app.run(debug=True)
