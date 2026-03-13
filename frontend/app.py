"""Virtual Try-On System - Streamlit Frontend"""

import tempfile
from pathlib import Path

import cv2
import streamlit as st

from frontend.auth import (
    authenticate_user,
    create_user,
    init_auth_db,
    initialize_auth_session,
    is_session_valid,
    login_session,
    logout_session,
    request_password_reset,
    reset_password_with_token,
)
from src.garment_manager import list_available_garments, load_garment_image, load_garment_metadata
from src.image_utils import load_image
from src.measurement_inference import infer_measurements, validate_measurements
from src.model_layer import load_models
from src.pose_detection import detect_pose
from src.segmentation import segment_body
from src.size_recommendation import explain_recommendation, recommend_size
from src.validation import validate_image


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Virtual Try-On",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .measurement-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def render_auth_page() -> None:
    """Render login, registration, and password reset forms."""
    st.markdown('<p class="title">Secure Access</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Login or create an account to use the Virtual Try-On system</p>',
        unsafe_allow_html=True,
    )
    login_tab, register_tab, forgot_tab = st.tabs(["Login", "Register", "Forgot Password"])

    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            login_id = st.text_input("Email or Username")
            password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login")

        if login_submit:
            success, message, user = authenticate_user(login_id, password)
            if success and user is not None:
                login_session(st.session_state, user)
                st.success("Login successful.")
                st.rerun()
            else:
                st.error(message)

    with register_tab:
        with st.form("register_form", clear_on_submit=True):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_submit = st.form_submit_button("Create Account")

        if register_submit:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = create_user(new_username, new_email, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    with forgot_tab:
        st.caption("Development mode: reset token is shown once here instead of email delivery.")
        with st.form("forgot_password_form", clear_on_submit=False):
            forgot_login_id = st.text_input("Email or Username", key="forgot_login_id")
            forgot_submit = st.form_submit_button("Generate Reset Token")

        if forgot_submit:
            success, message, reset_token = request_password_reset(forgot_login_id)
            if success:
                st.info(message)
                if reset_token:
                    st.code(reset_token)
                    st.warning("This token expires in 15 minutes.")
            else:
                st.error(message)

        with st.form("reset_password_form", clear_on_submit=True):
            reset_token_input = st.text_input("Reset Token")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            reset_submit = st.form_submit_button("Reset Password")

        if reset_submit:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = reset_password_with_token(reset_token_input, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)


# ============================================================================
# SIDEBAR
# ============================================================================

init_auth_db()
initialize_auth_session(st.session_state)
session_valid = is_session_valid(st.session_state)

st.sidebar.markdown("# Virtual Try-On System")
st.sidebar.markdown("---")

if session_valid:
    user_info = st.session_state.get("auth_user") or {}
    st.sidebar.success(f"Logged in as {user_info.get('username', 'user')}")
    if st.sidebar.button("Logout"):
        logout_session(st.session_state)
        st.rerun()

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Select Page:",
        ["Upload & Measure", "Try-On", "Garments"],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**How it works:**\n\n"
        "1. Upload a photo of yourself\n"
        "2. We detect your body and pose\n"
        "3. We infer your measurements\n"
        "4. We recommend clothing sizes\n"
        "5. Try on different garments!"
    )
else:
    page = None
    st.sidebar.info("Please login to use the application.")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_ai_models():
    """Load ML models once"""
    try:
        seg_model, pose_model = load_models()
        return seg_model, pose_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


def process_user_image(image_path):
    """Process user image: detect pose, segment body, infer measurements"""
    try:
        seg_model, pose_model = load_ai_models()

        if seg_model is None or pose_model is None:
            st.error("Models not loaded")
            return None

        image = load_image(image_path)

        with st.spinner("Segmenting body..."):
            seg_result = segment_body(image, seg_model)

        with st.spinner("Detecting pose..."):
            try:
                pose_result = detect_pose(image, pose_model)
            except RuntimeError as e:
                st.error(f"Pose detection failed: {e}")
                return None

        with st.spinner("Inferring measurements..."):
            measurements = infer_measurements(pose_result, seg_result)

        is_valid, errors = validate_measurements(measurements)

        if not is_valid:
            st.error(f"Measurement validation failed: {errors}")
            return None

        return {
            "image": image,
            "measurements": measurements,
            "pose": pose_result,
            "segmentation": seg_result,
        }

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


# ============================================================================
# PAGES
# ============================================================================

if not session_valid:
    render_auth_page()

elif page == "Upload & Measure":
    st.markdown('<p class="title">Upload & Measure</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a photo to measure your body</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Photo")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of yourself from the front",
        )

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            validation = validate_image(tmp_path)

            if not validation.is_valid:
                st.error("Image validation failed")
                for error in validation.errors:
                    st.write(f"- {error}")
            else:
                if validation.warnings:
                    st.warning("Warnings:")
                    for warning in validation.warnings:
                        st.write(f"- {warning}")

                image = load_image(tmp_path)
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="Uploaded image",
                    use_column_width=True,
                )

                if st.button("Analyze Photo", key="analyze_btn"):
                    result = process_user_image(tmp_path)

                    if result:
                        st.session_state.result = result
                        st.session_state.temp_path = tmp_path
                        st.success("Image processed successfully!")

    with col2:
        st.subheader("Your Measurements")

        if "result" in st.session_state:
            result = st.session_state.result
            measurements = result["measurements"]
            pose = result["pose"]

            st.markdown("#### Body Measurements")
            col_m1, col_m2 = st.columns(2)

            with col_m1:
                st.metric("Shoulder Width", f"{measurements.shoulder_width_cm:.1f} cm")
                st.metric("Torso Length", f"{measurements.torso_length_cm:.1f} cm")

            with col_m2:
                st.metric("Chest Circumference", f"{measurements.chest_circumference_cm:.1f} cm")
                st.metric("Measurement Confidence", f"{measurements.confidence * 100:.1f}%")

            st.markdown("#### Pose Analysis")
            st.write(f"**Is Frontal:** {'Yes' if pose.is_frontal else 'No'}")
            st.write(f"**Shoulder Width (px):** {pose.shoulder_width_px:.1f}")
            st.write(f"**Keypoints Detected:** {len(pose.keypoints)}")

            if pose.warnings:
                st.warning("**Pose Warnings:**")
                for warning in pose.warnings:
                    st.write(f"- {warning}")
        else:
            st.info("Upload and analyze a photo to see measurements")

elif page == "Try-On":
    st.markdown('<p class="title">Virtual Try-On</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">See how garments fit you</p>', unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.warning("Please upload and analyze a photo first!")
        st.stop()

    result = st.session_state.result
    measurements = result["measurements"]

    garments = list_available_garments()

    if not garments:
        st.error("No garments available")
        st.stop()

    st.subheader("Select a Garment")
    selected_garment = st.selectbox("Choose garment:", garments)

    try:
        metadata = load_garment_metadata(selected_garment)
    except FileNotFoundError:
        st.error(f"Garment not found: {selected_garment}")
        st.stop()

    size_chart = metadata.get("size_chart", {})

    if size_chart:
        recommendation = recommend_size(measurements, size_chart)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Garment Details")
            st.write(f"**Name:** {metadata.get('name', 'N/A')}")
            st.write(f"**Brand:** {metadata.get('brand', 'N/A')}")
            st.write(f"**Category:** {metadata.get('category', 'N/A')}")
            st.write(f"**Material:** {metadata.get('material', 'N/A')}")
            st.write(f"**Price:** ${metadata.get('price_usd', 0):.2f}")

            colors = metadata.get("available_colors", [])
            if colors:
                st.write(f"**Colors:** {', '.join(colors)}")

        with col2:
            st.markdown("#### Size Recommendation")

            st.metric(
                "Recommended Size",
                recommendation.size,
                f"{recommendation.confidence * 100:.1f}% confidence",
            )

            st.write("**Fit Scores by Size:**")
            for size in sorted(recommendation.fit_scores.keys()):
                score = recommendation.fit_scores[size]
                percentage = score * 100
                st.write(f"{size}: {percentage:.1f}%")

            st.markdown("#### Recommendation Details")
            explanation = explain_recommendation(recommendation, measurements)
            st.text(explanation)

        st.markdown("#### Size Chart")
        size_chart_data = []
        for size in sorted(size_chart.keys()):
            measurements_data = size_chart[size]
            size_chart_data.append(
                {
                    "Size": size,
                    "Shoulder (cm)": measurements_data.get("shoulder_width_cm", "N/A"),
                    "Chest (cm)": measurements_data.get("chest_circumference_cm", "N/A"),
                    "Torso (cm)": measurements_data.get("torso_length_cm", "N/A"),
                }
            )

        st.dataframe(size_chart_data, use_container_width=True)
    else:
        st.error("Size chart not available for this garment")

elif page == "Garments":
    st.markdown('<p class="title">Browse Garments</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Explore available clothing items</p>', unsafe_allow_html=True)

    garments = list_available_garments()

    if not garments:
        st.error("No garments available")
        st.stop()

    st.subheader(f"Available Garments ({len(garments)})")

    cols = st.columns(3)

    for idx, garment_id in enumerate(garments):
        try:
            metadata = load_garment_metadata(garment_id)

            with cols[idx % 3]:
                st.markdown(f"### {metadata.get('name', garment_id)}")

                try:
                    garment_img = load_garment_image(garment_id)
                    st.image(
                        cv2.cvtColor(garment_img, cv2.COLOR_BGR2RGB),
                        use_column_width=True,
                        caption=metadata.get("name", garment_id),
                    )
                except Exception:
                    st.info("No image available")

                st.write(f"**Brand:** {metadata.get('brand', 'N/A')}")
                st.write(f"**Category:** {metadata.get('category', 'N/A')}")
                st.write(f"**Price:** ${metadata.get('price_usd', 0):.2f}")

                colors = metadata.get("available_colors", [])
                if colors:
                    st.write(f"**Colors:** {', '.join(colors)}")

                if st.button(f"Try {metadata.get('name', 'this')}", key=garment_id):
                    st.session_state.selected_garment = garment_id
                    st.switch_page("pages/try_on.py") if Path("pages/try_on.py").exists() else None

        except Exception as e:
            st.error(f"Error loading {garment_id}: {e}")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>Virtual Try-On System v0.1.0</p>
        <p>Built with Streamlit, OpenCV, and MediaPipe</p>
    </div>
    """,
    unsafe_allow_html=True,
)
