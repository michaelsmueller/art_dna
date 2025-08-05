# === Import required libraries ===
import os  # Used to read environment variables
import streamlit as st  # Main library to build the web app UI
import requests  # To send HTTP requests to the backend API
import io  # To read uploaded image data as bytes
from dotenv import load_dotenv  # Loads variables from a .env file (e.g., USE_GCS)
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit.components.v1 as components

def plotly_rgb_to_hex(rgb_str):
    rgb = rgb_str.strip("rgb()").split(",")
    return "#{:02x}{:02x}{:02x}".format(*[int(float(x)) for x in rgb])

def genre_to_color_map(genres: list[str], colorscale="Turbo") -> dict:
    n = len(genres)
    positions = [i / (n - 1) if n > 1 else 0.5 for i in range(n)]
    colors = sample_colorscale(colorscale, positions, colortype="rgb")
    return dict(zip(genres, colors))

def get_chart_data(predictions: dict, top_k: int):
    top_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_preds_sorted = sorted(top_preds, key=lambda x: x[0])
    labels = [genre for genre, _ in top_preds_sorted]
    r = [float(score) for _, score in top_preds_sorted]
    n = len(labels)
    angles = np.linspace(0, 360, n, endpoint=False).tolist()
    max_r = max(r)
    return labels, r, angles, max_r

def radar_barpolar(predictions, top_k=6):
    labels, r, angles, max_r = get_chart_data(predictions, top_k)
    hover_text = [f"<span style='font-size:20px'><b>{round(score * 100)}%</b></span>" for score in r]

    color_map = genre_to_color_map(labels, colorscale="Turbo")
    rgb_colors = [color_map[genre] for genre in labels]

    fig = go.Figure(go.Barpolar(
        r=r,
        theta=angles,
        width=[(360 / len(labels)) * 0.3] * len(labels),
        marker=dict(color=rgb_colors, line=dict(width=1)),
        text=hover_text,
        hoverinfo="text",
        opacity=0.9
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, showticklabels=False, showline=False, ticks=''),
            angularaxis=dict(
                tickvals=angles,
                ticktext=labels,
                tickfont=dict(size=24),
                rotation=60,
                direction="clockwise",
                showline=False,
                showticklabels=True,
                ticks=''
            )
        ),
        showlegend=False,
        dragmode=False,
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

    return color_map

def fetch_genre_descriptions(genres: list[str], audience: str = "adult"):
    metadata = {}
    try:
        res = requests.get(
            "https://art-dna-api-521843227251.europe-west1.run.app/describe",
            params={"genres": ",".join(genres), "audience": audience},
            timeout=5
        )

        if res.ok:
            data = res.json()
            for item in data.get("descriptions", []):
                metadata[item["genre"]] = {
                    "description": item.get("description", ""),
                    "time_period": item.get("time_period", ""),
                    "key_artists": item.get("key_artists", []),
                    "visual_elements": item.get("visual_elements", []),
                    "philosophy": item.get("philosophy", "")
                }
        else:
            for genre in genres:
                metadata[genre] = {"description": "Failed to fetch description."}
    except Exception:
        for genre in genres:
            metadata[genre] = {"description": "Error fetching description."}
    return metadata



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

# === App version displayed at the bottom of the UI ===
APP_VERSION = "v1.0.2"

# === Configure Streamlit page layout and title ===
st.set_page_config(layout="wide")  # Use full width layout
st.title("Art Style Explorer 🎨")  # Main app title
st.set_page_config(layout="wide")  # Use full width layout
st.markdown(
    "<h4 style='font-size: 1.4rem; font-weight: 400;'>A dialogue between artists across centuries</h4>",
    unsafe_allow_html=True,
)

# === Optional debug mode checkbox ===
debug_mode = st.checkbox("Debug Mode")
kid_mode = st.checkbox("Kid Version")

# === File upload widget ===
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Store file on upload
if uploaded:
    st.session_state["uploaded_file"] = uploaded
    st.session_state["image_bytes"] = uploaded.read()
    st.session_state["image_name"] = uploaded.name
    st.session_state["image_type"] = uploaded.type

# Restore from session_state
uploaded_file = st.session_state.get("uploaded_file")
img_bytes = st.session_state.get("image_bytes")
img_name = st.session_state.get("image_name")
img_type = st.session_state.get("image_type")

# === MAIN LAYOUT ===
if uploaded_file:
    left_col, right_col = st.columns(2)

    with left_col:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with right_col:
        # Prediction trigger
        if st.button("Explore the artistic DNA of your image"):
            result, used_url = send_image_to_api(
                img_bytes,
                img_name,
                img_type,
                API_URL_PRIMARY,
                API_URL_FALLBACK,
            )

            if result and result.get("predictions"):
                st.session_state["predictions"] = result["predictions"]
                st.session_state["used_url"] = used_url
            else:
                st.error("No predictions found.")
                st.stop()

            sim_result, used_sim_url = send_image_to_api(
                img_bytes,
                img_name,
                img_type,
                SIMILARITY_API_URL_PRIMARY,
                SIMILARITY_API_URL_FALLBACK,
            )

            if sim_result:
                st.session_state["similar_images"] = sim_result.get("similar_images", [])
                st.session_state["used_sim_url"] = used_sim_url

        # === DISPLAY PREDICTIONS (inside right column) ===
        if "predictions" in st.session_state:
            predictions = st.session_state["predictions"]
            st.markdown("## Art Style Connections:")
            genre_colors = radar_barpolar(predictions, top_k=6)
            audience = "kid" if kid_mode else "adult"
            descriptions = fetch_genre_descriptions(list(genre_colors.keys()), audience=audience)

            html = f"""
            <div class='audience-{audience}' style="font-family:sans-serif;">
                <div id='hover_result' style='font-size:16px;color:#333;padding:10px;
                    min-height:160px; max-height:200px; overflow-y:auto; background-color:#fafafa;
                    border:1px solid #ccc; border-radius:6px; margin-bottom:30px;'></div>
                <div style="display:flex; flex-wrap:wrap; gap:20px;">
            """

            for genre, color in genre_colors.items():
                data = descriptions.get(genre, {})
                desc = data.get("description", "No description available.").replace("'", "\\'").replace('"', "&quot;")
                time_period = data.get("time_period", "Unknown period")
                philosophy = data.get("philosophy", "")
                artists = ", ".join(data.get("key_artists", []))
                elements = ", ".join(data.get("visual_elements", []))

                full_html = f"""
                <b>Description:</b> {desc}<br>
                <b>Time period:</b> {time_period}<br>
                <b>Key artists:</b> {artists}<br>
                <b>Visual elements:</b> {elements}<br>
                <b>Philosophy:</b> {philosophy}
                """.replace("'", "\\'").replace('"', "&quot;").replace("\n", "").replace("\r", "")

                html += f"""
                <div
                    onmouseover="this.style.backgroundColor='#e0e0ff'; document.getElementById('hover_result').innerHTML = '{full_html}';"
                    onmouseout="this.style.backgroundColor='#f0f0f0'; document.getElementById('hover_result').innerHTML = '';"
                    style="padding:8px 16px; border:1px solid #888; border-radius:5px;
                    background-color:#f0f0f0; color:{color}; font-size:16px; cursor:pointer;">
                    {genre}
                </div>
                """

            html += "</div></div>"
            components.html(html, height=350)

        # === DISPLAY SIMILAR IMAGES (still inside right_col) ===
        if "similar_images" in st.session_state:
            similar_images = st.session_state["similar_images"]
            if similar_images:
                st.markdown("---")
                st.markdown("## Shared Visual Features:")
                cols = st.columns(5)
                for idx, sim_img in enumerate(similar_images):
                    with cols[idx % 5]:
                        st.image(sim_img.get("image_url"), use_container_width=True)
                        artist = sim_img.get("artist_name", "Unknown Artist")
                        score = sim_img.get("similarity_score", 0)
                        st.caption(f"{artist}\nScore: {score:.2f}")

        # === DEBUG INFO ===
        if debug_mode:
            st.divider()
            st.markdown("Debug Info - Style Prediction API")
            st.write("API Used:", st.session_state.get("used_url"))
            st.json(st.session_state.get("predictions"))

            st.markdown("Debug Info - Similarity API")
            st.write("API Used:", st.session_state.get("used_sim_url"))
            st.json(st.session_state.get("similar_images"))


# === FOOTER ===
st.markdown(f"<hr><small>App version: {APP_VERSION}</small>", unsafe_allow_html=True)
