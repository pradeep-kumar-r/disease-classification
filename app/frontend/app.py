import os
import requests
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Chicken Disease Classifier", layout="wide")

st.markdown("""
<style>
    .reportview-container {background: #f0f2f6}
    .sidebar .sidebar-content {background: #ffffff}
    .stButton>button {background-color: #4CAF50; color: white}
    .stAlert {border-left-color: #4CAF50}
</style>
""", unsafe_allow_html=True)

st.title("🐔 Chicken Fecal Disease Classifier")
st.markdown("Upload fecal images to detect Salmonella, New Castle Disease, or Coccidiosis")

def display_prediction(img, predictions):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(list(predictions.keys()), list(predictions.values()))
    ax.set_xlabel('Probability')
    ax.set_title('Disease Probabilities')

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, use_column_width=True)
    with col2:
        st.pyplot(fig)

    st.markdown("### Diagnosis Summary")
    max_disease = max(predictions, key=predictions.get)
    st.success(f"Most likely: **{max_disease}** ({predictions[max_disease]:.1%} probability)")

tab1 = st.tabs(["Single Image"])[0]

with tab1:
    uploaded_file = st.file_uploader(
        "Choose a JPG image",
        type=["jpg", "jpeg"],
        key="single"
    )

    if uploaded_file:
        # First read the image for display
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        if st.button("Analyze Single Image", key="single_btn"):
            with st.spinner("Processing..."):
                try:
                    # Reset file pointer to the beginning before sending
                    uploaded_file.seek(0)
                    
                    # Create a tuple with filename, file object, and content type
                    files = {
                        'file': (uploaded_file.name, uploaded_file, 'image/jpeg')
                    }
                    response = requests.post(
                        f"{API_URL}/predict",
                        files=files,
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        probabilities = result['predictions']['probabilities']
                        display_prediction(image, probabilities)

                        df = pd.DataFrame({
                            'Disease': list(probabilities.keys()),
                            'Probability': list(probabilities.values())
                        })
                        df['Probability'] = df['Probability'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(df[['Disease', 'Probability']].reset_index(drop=True))
                    else:
                        st.error(f"API Error: {response.json()['detail']}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Network error: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

st.sidebar.markdown("## About")
st.sidebar.info(
    "This AI-powered tool helps identify common chicken diseases "
    "from fecal images using deep learning models."
)