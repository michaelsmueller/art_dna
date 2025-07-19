import streamlit as st
import requests
import io

API_URL = "http://localhost:8000/predict"
APP_VERSION = "v1.0.0"

st.set_page_config(layout="wide")
st.title("Art Style Classifier")
st.markdown("Upload a painting image (JPEG/PNG). Click 'Predict Style' to predict the art style")

debug_mode = st.checkbox("Debug Mode")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    left_col, right_col = st.columns(2)

    with left_col:
        st.image(uploaded_file, caption=" Uploaded Image", use_container_width=True)

    with right_col:
        if st.button(" Predict genres"):
            st.write(" Sending image to API...")

            try:
                img_bytes = uploaded_file.read()
                files = {"image": (uploaded_file.name, io.BytesIO(img_bytes), uploaded_file.type)}

                response = requests.post(API_URL, files=files)

                # Always show top prediction if possible
                if response.status_code == 200 and response.headers["content-type"] == "application/json":
                    data = response.json()
                    predictions = data.get("predictions")

                    if not predictions:
                        st.error("No 'predictions' in API response.")
                    else:
                        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                        top_preds = sorted_preds[:5]

                        st.markdown("Top 5 Predictions:")
                        for genre, confidence in top_preds:
                            st.markdown(f"- **{genre}**: {round(confidence * 100)}%")

                        if debug_mode:
                            st.divider()
                            st.markdown("Debug Info")
                            st.write("Status Code:", response.status_code)
                            st.write("Full JSON:")
                            st.json(data)

                else:
                    st.error("API returned invalid content or status.")

            except Exception as e:
                st.error(f"Request failed: {e}")

# Footer version
st.markdown(f"<hr><small> App version: {APP_VERSION}</small>", unsafe_allow_html=True)
