# predictor/views.py

from django.shortcuts import render
import joblib
import numpy as np
import os

# --- Load the model and scaler globally when the server starts ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'xgboost_model.joblib')
scaler_path = os.path.join(BASE_DIR, 'scaler.joblib')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def predictor(request):
    """
    Handles form display and prediction logic. Prepares number ranges
    for the template context.
    """
    # Define the number ranges for the form's radio buttons
    context = {
        'range_0_5': range(6),
        'range_1_3': range(1, 4),
        'range_0_3': range(4),
        'range_0_21': range(22),
        'range_0_27': range(28),
        'range_0_30': range(31),
    }

    if request.method == 'POST':
        try:
            # Collect data from the submitted form
            raw_data = [
                int(request.POST.get('anxiety_level')),
                int(request.POST.get('self_esteem')),
                int(request.POST.get('mental_health_history')),
                int(request.POST.get('depression')),
                int(request.POST.get('headache')),
                int(request.POST.get('blood_pressure')),
                int(request.POST.get('sleep_quality')),
                int(request.POST.get('breathing_problem')),
                int(request.POST.get('noise_level')),
                int(request.POST.get('living_conditions')),
                int(request.POST.get('safety')),
                int(request.POST.get('basic_needs')),
                int(request.POST.get('academic_performance')),
                int(request.POST.get('study_load')),
                int(request.POST.get('teacher_student_relationship')),
                int(request.POST.get('future_career_concerns')),
                int(request.POST.get('social_support')),
                int(request.POST.get('peer_pressure')),
                int(request.POST.get('extracurricular_activities')),
                int(request.POST.get('bullying'))
            ]
            
            # Scale the data and make a prediction
            input_array = np.array(raw_data).reshape(1, -1)
            scaled_input = scaler.transform(input_array)
            prediction_code = model.predict(scaled_input)[0]
            
            # Map the result to a user-friendly string
            stress_levels = {0: "Low Stress", 1: "Medium Stress", 2: "High Stress"}
            result_text = stress_levels.get(prediction_code, "Prediction error")

            # Add the result to the context dictionary to display it
            context['result'] = result_text

        except (ValueError, TypeError):
            # Handle cases where input is missing or not a number
            context['error'] = 'Invalid input. Please ensure all fields are filled correctly.'
    
    # Render the page, passing the context (which includes ranges and possibly a result/error)
    return render(request, 'predictor/index.html', context)
