import numpy as np
from flask import Flask, jsonify, request, render_template
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the model with error handling
try:
    model = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    print("Error: model.pkl file not found. Please ensure the file is in the correct directory.")
except Exception as e:
    print(f"Error loading model.pkl: {e}")

# Route for home page
@app.route('/')
def home():
    return render_template("index.html")

# Route for prediction through form submission on HTML GUI
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering prediction on HTML GUI
    '''
    try:
        # Get input features from the form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Predict using the loaded model
        prediction = model.predict(final_features)
        
        # Determine the result based on the prediction
        if prediction[0] == 0:
            result = "breast cancer report is negative"
        else:
            result = "breast cancer report is positive"

        # Render the result on the index page
        return render_template("index.html", prediction_text=result)
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

# Route for API prediction
@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API call through request
    '''
    try:
        # Parse the incoming JSON data
        data = request.get_json(force=True)
        prediction = model.predict([np.array(list(data.values()))])
        
        # Format the output based on the prediction
        output = "breast cancer is positive" if prediction[0] == 1 else "breast cancer is negative"
        
        # Return the result as JSON
        return jsonify(output)
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

# Make sure templates folder is used
app = Flask(__name__, template_folder='templates')








