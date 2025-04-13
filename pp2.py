from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Paths to your data files
file_path = 'data.csv.xlsx'  # Path to rainfall data
crops_file_path = 'crops.xlsx'  # Path to crops data

# Load crops data
crops_data = pd.read_excel(crops_file_path)

# Function to load and preprocess the rainfall data
def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path, sheet_name='Sheet1')
    data = data.dropna(subset=['District', 'Taluk', 'Hobli', 'Annual'])
    
    label_encoders = {}
    for column in ['District', 'Taluk', 'Hobli']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le
    
    return data, label_encoders

# Function to train and save the model
def train_and_save_model(data, label_encoders):
    X = data[['District', 'Taluk', 'Hobli']]
    y = data['Annual']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'decision_tree_regressor.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')

# Load data and model
try:
    data, label_encoders = load_and_preprocess_data(file_path)
    train_and_save_model(data, label_encoders)
    model = joblib.load('decision_tree_regressor.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    unique_districts = label_encoders['District'].inverse_transform(data['District'].unique())
except Exception as e:
    print(f"Error loading data or training model: {e}")
    data, label_encoders, unique_districts = None, None, []


@app.route('/')
def welcome():
    return render_template('welcome.html', name="Rainfall Predictor")


@app.route('/form')
def form():
    return render_template('index2.html', districts=unique_districts)


@app.route('/get_taluks', methods=['POST'])
def get_taluks():
    selected_district = request.form['district']
    try:
        le_district = label_encoders['District']
        le_taluk = label_encoders['Taluk']
        selected_district_num = le_district.transform([selected_district])[0]
        unique_taluks = data[data['District'] == selected_district_num]['Taluk'].dropna().unique()
        unique_taluks = le_taluk.inverse_transform(unique_taluks)
        return jsonify({'taluks': unique_taluks.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/get_hoblis', methods=['POST'])
def get_hoblis():
    selected_district = request.form['district']
    selected_taluk = request.form['taluk']
    try:
        le_district = label_encoders['District']
        le_taluk = label_encoders['Taluk']
        le_hobli = label_encoders['Hobli']
        selected_district_num = le_district.transform([selected_district])[0]
        selected_taluk_num = le_taluk.transform([selected_taluk])[0]
        unique_hoblis = data[(data['District'] == selected_district_num) & 
                             (data['Taluk'] == selected_taluk_num)]['Hobli'].dropna().unique()
        unique_hoblis = le_hobli.inverse_transform(unique_hoblis)
        return jsonify({'hoblis': unique_hoblis.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    selected_district = request.form['district']
    selected_taluk = request.form['taluk']
    selected_hobli = request.form['hobli']
    try:
        # Encode the input data
        le_district = label_encoders['District']
        le_taluk = label_encoders['Taluk']
        le_hobli = label_encoders['Hobli']

        selected_district_num = le_district.transform([selected_district])[0]
        selected_taluk_num = le_taluk.transform([selected_taluk])[0]
        selected_hobli_num = le_hobli.transform([selected_hobli])[0]

        # Prepare input for prediction
        input_data = [[selected_district_num, selected_taluk_num, selected_hobli_num]]

        # Predict using the Decision Tree Regressor
        predicted_annual_rainfall = model.predict(input_data)[0]

        # Filter data based on the selections
        filtered_data = data[(data['District'] == selected_district_num) & 
                              (data['Taluk'] == selected_taluk_num) & 
                              (data['Hobli'] == selected_hobli_num)]

        if filtered_data.empty:
            return "Error: No data available for the selected location.", 400

        # Predict monthly rainfall using historical averages
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        predicted_monthly_rainfall = filtered_data[months].mean()

        # Calculate the average annual rainfall
        predicted_annual_rainfall = filtered_data['Annual'].mean()

        # Get the current month and day
        current_month = datetime.now().strftime("%b")
        current_day = datetime.now().day

        # Check if the current month's column exists in the dataset
        if current_month not in filtered_data.columns:
            predicted_today_rainfall = 0
        else:
            predicted_today_rainfall = filtered_data[current_month].mean()

        # Calculate the average daily rainfall for the current month
        average_daily_rainfall = predicted_today_rainfall / 30 if predicted_today_rainfall > 0 else 0

        # Prepare results
        results = {
            'monthly': predicted_monthly_rainfall.to_dict(),
            'annual': predicted_annual_rainfall,
            'today': average_daily_rainfall,
            'current_month': current_month,
            'current_day': current_day
        }

        # Generate pie and bar charts
        months = list(results['monthly'].keys())
        rainfall = list(results['monthly'].values())

        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie(rainfall, labels=months, autopct='%1.1f%%')
        pie_chart = io.BytesIO()
        plt.savefig(pie_chart, format='png')
        pie_chart.seek(0)
        pie_chart_base64 = base64.b64encode(pie_chart.getvalue()).decode('utf8')
        plt.close(fig)

        # Bar Chart
        fig, ax = plt.subplots()
        ax.bar(months, rainfall)
        bar_chart = io.BytesIO()
        plt.savefig(bar_chart, format='png')
        bar_chart.seek(0)
        bar_chart_base64 = base64.b64encode(bar_chart.getvalue()).decode('utf8')
        plt.close(fig)

        # Update the results dictionary with chart images
        results.update({
            'pie_chart': pie_chart_base64,
            'bar_chart': bar_chart_base64
        })

        # Return the result page with the selected places and results
        return render_template('result.html', 
                               results=results, 
                               district=selected_district,
                               taluk=selected_taluk,
                               hobli=selected_hobli)

    except Exception as e:
        return f"Error during prediction: {e}", 500


@app.route('/agriculture_analysis')
def agriculture_analysis():
    return render_template('agriculture_analysis.html')


@app.route('/process_agriculture_analysis', methods=['POST'])
def process_agriculture_analysis():
    try:
        # Get form input values
        ph_value = request.form.get('ph_value')
        soil_type = request.form.get('soil_type')

        if not ph_value or not soil_type:
            return "Both pH value and soil type are required.", 400

        ph_value = float(ph_value)
        crops_data.columns = crops_data.columns.str.strip()  # Clean up column names

        # Normalize the soil type for comparison
        soil_type = soil_type.strip().lower()

        # Filter crops based on soil type (case-insensitive)
        recommended_crops = crops_data[crops_data['Soil Type'].str.lower() == soil_type]

        if recommended_crops.empty:
            return "No crops found for the specified soil type.", 400

        # Filter based on pH value and rainfall
        if ph_value < 6.0:
            recommended_crops = recommended_crops[recommended_crops['Rainfall Min'] >= 1000]
        elif ph_value > 7.5:
            recommended_crops = recommended_crops[recommended_crops['Rainfall Min'] < 1000]

        # Soil pH analysis
        analysis = {
            'ph_value': ph_value,
            'moisture_level': "High" if soil_type == 'clay' else "Medium",  # Example, adjust based on actual input
            'soil_type': soil_type.capitalize(),
            'temperature_range': "20°C - 30°C",  # Example, adjust based on actual logic
            'recommendation': "Optimal pH for most crops" if 6.0 <= ph_value <= 7.5 else "Soil pH needs adjustment",
            'best_crops': recommended_crops.to_dict('records')
        }

        # Convert the recommended crops to a list of dictionaries for rendering
        crop_recommendations = recommended_crops.to_dict('records')

        # Pass analysis and crop details to the template
        return render_template('analysis_results.html', analysis=analysis, crops=crop_recommendations)

    except Exception as e:
        return f"Error during analysis: {e}", 400





if __name__ == '__main__':
    app.run(debug=True)