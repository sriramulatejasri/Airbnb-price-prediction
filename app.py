import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load the trained model
try:
    model_data = joblib.load("model.pkl")
    model = model_data['model']
    feature_columns = model_data['feature_columns']
    performance = model_data['performance']
    print("✅ Model loaded successfully!")
    print(f"📊 Model Performance - MAE: ${performance['test_mae']:.2f}, R²: {performance['test_r2']:.3f}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def predict_price(minimum_nights, number_of_reviews, availability_365, room_type, neighbourhood_group):
    """Predict Airbnb price based on input features"""
    if model is None:
        return "❌ Model not loaded properly. Please check the model file."

    try:
        # Create input dataframe
        input_data = pd.DataFrame([{
            'minimum_nights': minimum_nights,
            'number_of_reviews': number_of_reviews,
            'availability_365': availability_365,
            'room_type': room_type,
            'neighbourhood_group': neighbourhood_group
        }])

        # Preprocess input (same as training)
        input_encoded = pd.get_dummies(input_data,
                                       columns=['room_type', 'neighbourhood_group'],
                                       prefix=['room', 'neighbourhood'])

        # Ensure all expected columns are present
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reorder columns to match training data
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_encoded)[0]

        # Format output
        result = f"💸 *Estimated Price: ${prediction:.2f} per night*\n\n"
        result += "📋 *Input Summary:*\n"
        result += f"• Minimum nights: {minimum_nights}\n"
        result += f"• Number of reviews: {number_of_reviews}\n"
        result += f"• Availability: {availability_365} days/year\n"
        result += f"• Room type: {room_type}\n"
        result += f"• Neighbourhood: {neighbourhood_group}\n\n"
        result += f"📈 *Model Performance:* MAE ${performance['test_mae']:.2f}, R² {performance['test_r2']:.3f}"
        return result

    except Exception as e:
        return f"❌ Error making prediction: {str(e)}"

# Define the Gradio interface
def create_interface():
    """Create and configure the Gradio interface"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏠 Airbnb Price Predictor
        *Predict nightly prices for Airbnb listings using machine learning!*
        This app uses a Random Forest model trained on Airbnb listing features to estimate
        optimal pricing. Simply enter your property details below to get a price prediction.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛 Property Details")

                minimum_nights = gr.Slider(
                    minimum=1, maximum=30, value=2, step=1,
                    label="Minimum Nights",
                    info="Minimum number of nights guests must book"
                )

                number_of_reviews = gr.Slider(
                    minimum=0, maximum=100, value=15, step=1,
                    label="Number of Reviews",
                    info="Total reviews received by the listing"
                )

                availability_365 = gr.Slider(
                    minimum=0, maximum=365, value=200, step=1,
                    label="Availability (days/year)",
                    info="Number of days the property is available for booking"
                )

                room_type = gr.Dropdown(
                    choices=["Entire home", "Private room", "Shared room"],
                    value="Entire home",
                    label="Room Type",
                    info="Type of accommodation offered"
                )

                neighbourhood_group = gr.Dropdown(
                    choices=["Manhattan", "Brooklyn", "Queens", "Bronx"],
                    value="Manhattan",
                    label="Neighbourhood Group",
                    info="General area where the property is located"
                )

                predict_btn = gr.Button("🔮 Predict Price", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Price Prediction")
                output = gr.Textbox(
                    label="Prediction Result",
                    placeholder="Enter property details and click 'Predict Price' to see the estimated nightly rate...",
                    lines=12,
                    max_lines=12
                )

        # Connect the prediction function
        predict_btn.click(
            fn=predict_price,
            inputs=[minimum_nights, number_of_reviews, availability_365, room_type, neighbourhood_group],
            outputs=output
        )

        gr.Markdown("""
        ---
        ### 🎯 How it Works
        This predictor uses a *Random Forest* algorithm trained on Airbnb listing data including:
        - Property characteristics (room type, availability)
        - Location factors (neighbourhood group)
        - Social proof (number of reviews)
        - Booking requirements (minimum nights)

        The model learns complex patterns in pricing to provide accurate estimates for new listings.

        ### 📈 Model Performance
        - *Mean Absolute Error (MAE)*: Average prediction error in dollars
        - *R² Score*: Proportion of price variance explained by the model

        Note: This is a demonstration model trained on simulated data for educational purposes.
        """)

    return demo

# ✅ App launch entry point
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)