import pickle
from flask import Flask, request, render_template

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Load input encoders
input_encoders = pickle.load(open('input_encoders.pkl', 'rb'))


# Optional: load output encoder
try:
    stage_encoder = pickle.load(open('stage_encoder.pkl', 'rb'))
except:
    stage_encoder = None

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/contact')
def contact():
    return render_template('details.html')

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define expected input fields
        expected_fields = [
            'Gender', 'Age', 'History', 'Patient', 'TakeMedication',
            'Severity', 'BreathShortness', 'VisualChanges', 'NoseBleeding',
            'Whendiagnoused', 'Systolic', 'Diastolic', 'ControlledDiet'
        ]

        # Validate and collect form data
        form_data = {}
        missing_fields = []
        for field in expected_fields:
            value = request.form.get(field)
            if value is None:
                missing_fields.append(field)
            else:
                form_data[field] = value

        if missing_fields:
            return render_template('result.html', prediction=f"❌ Missing fields: {', '.join(missing_fields)}")

        # Encode input
        encoded_input = []
        for field in expected_fields:
            if field not in input_encoders:
                return render_template('result.html', prediction=f"❌ Encoder missing for: {field}")
            encoder = input_encoders[field]
            encoded_value = encoder.transform([form_data[field]])[0]
            encoded_input.append(encoded_value)

        # Debug print
        print("Encoded input:", encoded_input)

        # Predict
        if len(encoded_input) != 13:
            return render_template('result.html', prediction=f"❌ Error: Expected 13 features, got {len(encoded_input)}")

        prediction = model.predict([encoded_input])[0]

        # Decode prediction if needed
        if stage_encoder:
            predicted_stage = stage_encoder.inverse_transform([prediction])[0]
        else:
            predicted_stage = prediction

        return render_template('result.html', prediction=f"✅ Predicted Stage: {predicted_stage}")

    except Exception as e:
        return render_template('result.html', prediction=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
