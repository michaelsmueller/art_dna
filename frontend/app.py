import streamlit as st
import requests
import io

API_URL_PRIMARY = "https://art-dna-api-521843227251.europe-west1.run.app/"
API_URL_FALLBACK = "http://localhost:8000/predict"
APP_VERSION = "v1.0.0"

st.set_page_config(layout="wide")
st.title("Art Style Classifier")
st.markdown("Upload a painting image (JPEG/PNG). Click 'Predict Style' to predict the art style")

debug_mode = st.checkbox("Debug Mode")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def send_image_to_api(image_bytes, filename, mime):
    files = {"image": (filename, io.BytesIO(image_bytes), mime)}
    for url in [API_URL_PRIMARY, API_URL_FALLBACK]:
        try:
            response = requests.post(url, files=files, timeout=10)
            if response.status_code == 200 and response.headers.get("content-type") == "application/json":
                return response.json(), url
        except Exception as e:
            if debug_mode:
                st.warning(f"Failed to reach {url}: {e}")
    return None, None

if uploaded_file is not None:
    left_col, right_col = st.columns(2)

    with left_col:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with right_col:
        if st.button("Predict genres"):
            st.write("Sending image to API...")

            img_bytes = uploaded_file.read()
            result, used_url = send_image_to_api(img_bytes, uploaded_file.name, uploaded_file.type)

            if result is not None:
                predictions = result.get("predictions")
                if predictions:
                    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
                    st.markdown("Top 5 Predictions:")
                    for genre, confidence in sorted_preds:
                        st.markdown(f"- **{genre}**: {round(confidence * 100)}%")

                    if debug_mode:
                        st.divider()
                        st.markdown("Debug Info")
                        st.write("API Used:", used_url)
                        st.json(result)
                else:
                    st.error("No predictions found in the API response.")
            else:
                st.error("Both API endpoints failed.")

# Footer version
st.markdown(f"<hr><small>App version: {APP_VERSION}</small>", unsafe_allow_html=True)
