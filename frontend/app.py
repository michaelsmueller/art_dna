# === Import required libraries ===
import os  # Used to read environment variables
import streamlit as st  # Main library to build the web app UI
import requests  # To send HTTP requests to the backend API
import io  # To read uploaded image data as bytes
from dotenv import load_dotenv  # Loads variables from a .env file (e.g., USE_GCS)

# === Load environment variables ===
load_dotenv()

# === Set up API endpoints for style prediction ===
if os.getenv("USE_GCS", "false").lower() == "true":
    # Running in production (cloud deployment)
    API_URL_PRIMARY = "https://art-dna-api-521843227251.europe-west1.run.app/predict"
    API_URL_FALLBACK = "http://localhost:8000/predict"
else:
    # Running locally (during development)
    API_URL_PRIMARY = "http://localhost:8000/predict"
    API_URL_FALLBACK = "https://art-dna-api-521843227251.europe-west1.run.app/predict"

# === Set up API endpoints for similarity search ===
if os.getenv("USE_GCS", "false").lower() == "true":
    SIMILARITY_API_URL_PRIMARY = (
        "https://art-dna-api-521843227251.europe-west1.run.app/similar"
    )
    SIMILARITY_API_URL_FALLBACK = "http://localhost:8000/similar"
else:
    SIMILARITY_API_URL_PRIMARY = "http://localhost:8000/similar"
    SIMILARITY_API_URL_FALLBACK = (
        "https://art-dna-api-521843227251.europe-west1.run.app/similar"
    )

# === App version displayed at the bottom of the UI ===
APP_VERSION = "v1.0.1"

# === Configure Streamlit page layout and title ===
st.set_page_config(layout="wide")  # Use full width layout
st.title("Art Style Explorer 🎨")  # Main app title
st.set_page_config(layout="wide")  # Use full width layout
st.markdown(
    "<h4 style='font-size: 1.4rem; font-weight: 400;'>A dialogue between artists across centuries</h4>",
    unsafe_allow_html=True,
)

# === Optional debug mode checkbox ===
debug_mode = st.checkbox("Debug Mode")  # Shows detailed API messages if checked

# === File upload widget (supports only image files) ===
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# === Reusable function to send image to any API (style or similarity) ===
def send_image_to_api(image_bytes, filename, mime, primary_url, fallback_url):
    files = {
        "image": (filename, io.BytesIO(image_bytes), mime)
    }  # Package image as a file
    for url in [primary_url, fallback_url]:  # Try primary first, then fallback
        try:
            response = requests.post(url, files=files, timeout=10)  # Send the image
            if (
                response.status_code == 200
                and response.headers.get("content-type") == "application/json"
            ):
                return response.json(), url  # Return result if successful
        except Exception as e:
            if debug_mode:
                st.warning(
                    f"Failed to reach {url}: {e}"
                )  # Show warning if request fails
    return None, None  # Return nothing if both fail


# === Main logic: if user uploaded an image ===
if uploaded_file is not None:
    left_col, right_col = st.columns(2)  # Split layout: image on left, results on right

    with left_col:
        st.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )  # Show uploaded image

    with right_col:
        if st.button(
            "Explore the artistic DNA of your image"
        ):  # When user clicks "Predict genres"
            # st.write("Analysing the art style...")

            img_bytes = uploaded_file.read()  # Read image as bytes

            # === STYLE PREDICTION API CALL ===
            result, used_url = send_image_to_api(
                img_bytes,
                uploaded_file.name,
                uploaded_file.type,
                API_URL_PRIMARY,
                API_URL_FALLBACK,
            )

            if result is not None:
                predictions = result.get("predictions")
                if predictions:
                    # Sort and show top 5 predicted genres
                    sorted_preds = sorted(
                        predictions.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    st.markdown("## Art Style Connections:")
                    for genre, confidence in sorted_preds:
                        st.markdown(f"- **{genre}**: {round(confidence * 100)}%")
                else:
                    st.error("No predictions found in the API response.")
            else:
                st.error("Both prediction API endpoints failed.")
                st.stop()  # Stop further execution if predictions fail

            # === SIMILARITY API CALL ===
            st.markdown("---")
            # st.write("Searching for visually similar artworks...")

            similar_result, used_sim_url = send_image_to_api(
                img_bytes,
                uploaded_file.name,
                uploaded_file.type,
                SIMILARITY_API_URL_PRIMARY,
                SIMILARITY_API_URL_FALLBACK,
            )

            if similar_result is not None:
                similar_images = similar_result.get("similar_images", [])
                if similar_images:
                    st.markdown("## Shared Visual Features:")
                    cols = st.columns(5)  # 5 images per row
                    for idx, sim_img in enumerate(similar_images):
                        with cols[idx % 5]:  # Rotate through columns
                            img_url = sim_img.get("image_url")
                            artist = sim_img.get("artist_name", "Unknown Artist")
                            score = sim_img.get("similarity_score", 0)
                            st.image(img_url, use_container_width=True)
                            st.caption(f"{artist}\nScore: {score:.2f}")
                else:
                    st.info("No similar images found.")
            else:
                st.error("Failed to fetch similar images.")

            # === Optional debug information ===
            if debug_mode:
                st.divider()
                st.markdown("Debug Info - Style Prediction API")
                st.write("API Used:", used_url)
                st.json(result)

                st.markdown("Debug Info - Similarity API")
                st.write("API Used:", used_sim_url)
                st.json(similar_result)

# === Footer ===
st.markdown(f"<hr><small>App version: {APP_VERSION}</small>", unsafe_allow_html=True)
