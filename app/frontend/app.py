import streamlit as st
import requests
import json
import pandas as pd
from PIL import Image
import io
import os

# Configure the page
st.set_page_config(
    page_title="Disease Classification",
    page_icon="🏥",
    layout="wide"
)

# Constants
API_URL = "http://backend:8000"  # This will be the service name in Kubernetes

def main():
    st.title("Disease Classification App")
    st.write("Upload an image to classify the disease")

    # Create tabs for single and bulk upload
    tab1, tab2 = st.tabs(["Single Image", "Bulk Upload"])

    with tab1:
        st.header("Single Image Upload")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                try:
                    # Prepare the file for API request
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    
                    # Make API request
                    response = requests.post(f"{API_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        predictions = response.json()
                        
                        # Create a DataFrame for better visualization
                        df = pd.DataFrame({
                            'Disease': list(predictions.keys()),
                            'Probability': list(predictions.values())
                        })
                        df['Probability'] = df['Probability'].apply(lambda x: f"{x*100:.2f}%")
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Display the highest probability prediction
                        max_prob = max(predictions.items(), key=lambda x: x[1])
                        st.success(f"Most likely disease: {max_prob[0]} ({max_prob[1]*100:.2f}%)")
                    else:
                        st.error("Error in prediction. Please try again.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    with tab2:
        st.header("Bulk Image Upload")
        uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            if st.button("Predict All"):
                try:
                    # Prepare files for API request
                    files = [("files", (file.name, file.getvalue())) for file in uploaded_files]
                    
                    # Make API request
                    response = requests.post(f"{API_URL}/predict_bulk", files=files)
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Display results for each image
                        for result in results:
                            st.subheader(f"Results for {result['filename']}")
                            
                            # Create DataFrame for this image
                            df = pd.DataFrame({
                                'Disease': list(result['predictions'].keys()),
                                'Probability': list(result['predictions'].values())
                            })
                            df['Probability'] = df['Probability'].apply(lambda x: f"{x*100:.2f}%")
                            
                            st.dataframe(df, use_container_width=True)
                            
                            # Display the highest probability prediction
                            max_prob = max(result['predictions'].items(), key=lambda x: x[1])
                            st.success(f"Most likely disease: {max_prob[0]} ({max_prob[1]*100:.2f}%)")
                            st.markdown("---")
                    else:
                        st.error("Error in prediction. Please try again.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 