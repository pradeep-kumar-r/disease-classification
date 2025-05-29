import io
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

tab1, tab2 = st.tabs(["Single Image", "Bulk Analysis"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose a JPG image",
        type=["jpg", "jpeg"],
        key="single"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        if st.button("Analyze Single Image", key="single_btn"):
            with st.spinner("Processing..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": uploaded_file},
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        display_prediction(image, result["predictions"])

                        df = pd.DataFrame({
                            'Disease': list(result['predictions'].keys()),
                            'Probability': list(result['predictions'].values())
                        })
                        df['Probability'] = df['Probability'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(df.style.format({'Probability': '{:.1%}'}), hide_index=True)
                    else:
                        st.error(f"API Error: {response.json()['detail']}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Network error: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

with tab2:
    st.markdown("### Bulk Image Analysis")
    uploaded_files = st.file_uploader(
        "Select multiple JPG images",
        type=["jpg", "jpeg"],
        accept_multiple_files=True,
        key="bulk"
    )

    if uploaded_files:
        if st.button("Analyze All Images", key="bulk_btn"):
            try:
                with st.spinner(f"Processing {len(uploaded_files)} images..."):
                    files = [("files", file) for file in uploaded_files]
                    response = requests.post(
                        f"{API_URL}/predict_bulk",
                        files=files,
                        timeout=60
                    )

                if response.status_code == 200:
                    results = response.json()
                    st.subheader("Bulk Analysis Report")

                    summary_data = []
                    for result in results:
                        if result["predictions"]:
                            max_disease = max(result["predictions"], key=result["predictions"].get)
                            summary_data.append({
                                "Filename": result["filename"],
                                "Primary Diagnosis": max_disease,
                                "Confidence": f"{result['predictions'][max_disease]:.1%}"
                            })

                    if summary_data:
                        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

                    st.markdown("### Detailed Results")
                    for result in results:
                        with st.expander(f"{result['filename']}"):
                            if result["predictions"]:
                                img_bytes = next(f[1].getvalue() for f in files if f[0] == result['filename'])
                                image = Image.open(io.BytesIO(img_bytes))
                                display_prediction(image, result["predictions"])
                            else:
                                st.error(f"Error: {result['error']}")
            except requests.exceptions.RequestException as e:
                st.error(f"Network error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

st.sidebar.markdown("## About")
st.sidebar.info(
    "This AI-powered tool helps identify common chicken diseases "
    "from fecal images using deep learning models."
)