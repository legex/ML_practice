from flask import Flask, request, jsonify
import joblib
import numpy as np
import xgboost as xgb

# Load saved model
model = joblib.load("ML_practice\Extrovert_Vs_Introvert\XGBoost_best_classification.joblib")  # update path if needed

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing "features" key in request body'}), 400
    
    features = data['features']
    
    try:
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        result = prediction[0]
        return jsonify({'prediction': str(result)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8443, debug=True)
