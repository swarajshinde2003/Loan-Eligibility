from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the trained model
with open("random3.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Function to predict loan approval
def predict_loan_approval(input_data):
    input_data_np = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(input_data_np)
    return "Approved" if prediction == 0 else "Rejected"

# Home route to display form
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input from the form
        no_of_dependents = int(request.form["no_of_dependents"])
        education = request.form["education"]
        self_employed = request.form["self_employed"]
        income_annum = int(request.form["income_annum"])
        loan_amount = int(request.form["loan_amount"])
        loan_term = int(request.form["loan_term"])
        cibil_score = int(request.form["cibil_score"])
        residential_assets_value = int(request.form["residential_assets_value"])
        commercial_assets_value = int(request.form["commercial_assets_value"])
        luxury_assets_value = int(request.form["luxury_assets_value"])
        bank_asset_value = int(request.form["bank_asset_value"])

        # Encoding categorical inputs
        education_encoded = 0 if education == "Graduate" else 1
        self_employed_encoded = 0 if self_employed == "No" else 1

        # Input data
        input_data = {
            'no_of_dependents': no_of_dependents,
            'education': education_encoded,
            'self_employed': self_employed_encoded,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets_value,
            'commercial_assets_value': commercial_assets_value,
            'luxury_assets_value': luxury_assets_value,
            'bank_asset_value': bank_asset_value
        }

        # Get prediction result
        result = predict_loan_approval(input_data)
        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
