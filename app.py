import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from pathlib import Path


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="EstateAI | Smart Property Valuation",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "house_price_model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# ==========================================================
# 2. LOAD MODEL AND SCALER
# ==========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
    scaler = load_scaler()
    artifacts_loaded = True

except Exception as error:
    artifacts_loaded = False
    st.error(f"Model loading failed: {error}")


# ==========================================================
# 3. CUSTOM WEBSITE DESIGN
# ==========================================================

st.markdown("""
<style>

/* ---------------- GLOBAL ---------------- */

.stApp {
    background:
        linear-gradient(
            rgba(250, 248, 242, 0.84),
            rgba(250, 248, 242, 0.91)
        ),
        url("https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?auto=format&fit=crop&w=2200&q=90");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Hide default Streamlit elements */

header {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* ---------------- NAVIGATION ---------------- */

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 18px 30px;

    background: rgba(255,255,255,0.97);

    border: 1px solid #e9e5da;
    border-radius: 16px;

    box-shadow: 0 6px 25px rgba(35,50,35,0.07);
}

.brand {
    font-size: 27px;
    font-weight: 850;
    color: #194c38;
    letter-spacing: 1px;
}

.brand span {
    color: #c77d4b;
}

.nav-text {
    color: #647568;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}

/* ---------------- HERO ---------------- */

.hero {
    padding: 65px 10px 35px 10px;
}

.hero-badge {
    display: inline-block;

    background: #e7efe5;
    color: #285c43;

    padding: 10px 18px;
    border-radius: 30px;

    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.2px;
}

.hero h1 {
    font-size: 58px;
    line-height: 1.15;
    font-weight: 850;

    color: #183e30;

    margin-top: 22px;
    margin-bottom: 15px;
}

.hero h1 span {
    color: #c77d4b;
}

.hero p {
    font-size: 17px;
    line-height: 1.8;
    color: #52645a;
    max-width: 700px;
}

/* ---------------- STAT CARDS ---------------- */

.stat-card {
    background: rgba(255,255,255,0.97);

    border: 1px solid #e9e5da;
    border-radius: 16px;

    padding: 23px 15px;

    text-align: center;

    box-shadow: 0 5px 20px rgba(35,50,35,0.05);
}

.stat-number {
    color: #235b41;
    font-size: 29px;
    font-weight: 850;
}

.stat-label {
    color: #68776c;
    font-size: 13px;
    font-weight: 600;
    margin-top: 5px;
}

/* ---------------- SECTION HEADER ---------------- */

.section-header {
    background: #ffffff;

    padding: 25px 30px;

    border-radius: 17px 17px 0 0;

    border: 1px solid #e7e4da;
    border-bottom: none;
}

.section-header h2 {
    color: #194b37;
    font-size: 25px;
    font-weight: 800;
    margin: 0;
}

.section-header p {
    color: #738075;
    font-size: 14px;
    margin-top: 8px;
    margin-bottom: 0;
}

/* ---------------- FORM CONTAINER ---------------- */

div[data-testid="stForm"] {
    background: #ffffff;

    padding: 28px 30px 30px 30px;

    border: 1px solid #e7e4da;
    border-radius: 0 0 17px 17px;

    box-shadow: 0 12px 35px rgba(35,50,35,0.07);
}

/* ---------------- INPUT LABELS ---------------- */

.stApp label {
    color: #294b39 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}

/* ---------------- WHITE INPUT BOXES ---------------- */

div[data-baseweb="input"] {
    background: #ffffff !important;
    border: 1px solid #d9ded5 !important;
    border-radius: 9px !important;
}

div[data-baseweb="input"]:focus-within {
    border: 1px solid #347354 !important;
    box-shadow: 0 0 0 2px rgba(52,115,84,0.12) !important;
}

div[data-baseweb="input"] input {
    background: #ffffff !important;
    color: #253b30 !important;
    -webkit-text-fill-color: #253b30 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

div[data-baseweb="input"] button {
    color: #286047 !important;
}

/* ---------------- HELP TEXT ---------------- */

.stTooltipIcon {
    color: #c77d4b !important;
}

/* ---------------- SUBMIT BUTTON ---------------- */

.stFormSubmitButton button {
    width: 100%;
    height: 55px;

    background: #245b41 !important;
    color: #ffffff !important;

    border: none !important;
    border-radius: 10px !important;

    font-size: 16px !important;
    font-weight: 800 !important;

    transition: all 0.25s ease;
}

.stFormSubmitButton button:hover {
    background: #19442f !important;
    transform: translateY(-2px);
}

/* ---------------- RESULT CARD ---------------- */

.result-card {
    background: linear-gradient(
        120deg,
        #194c38,
        #347452
    );

    border-radius: 20px;

    padding: 40px 25px;

    text-align: center;

    margin-top: 30px;

    box-shadow: 0 15px 40px rgba(25,76,56,0.20);
}

.result-tag {
    color: #dcecdf;
    font-size: 13px;
    font-weight: 750;
    letter-spacing: 1.5px;
}

.result-price {
    color: #ffffff;
    font-size: 48px;
    font-weight: 850;
    margin: 17px 0;
}

.result-note {
    color: #e0eee3;
    font-size: 14px;
}

/* ---------------- DETAILS ---------------- */

.detail-card {
    background: rgba(255,255,255,0.97);

    border: 1px solid #e9e5da;
    border-radius: 14px;

    padding: 20px;

    margin-top: 15px;
}

.detail-title {
    color: #194c38;
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 15px;
}

.detail-value {
    color: #52645a;
    font-size: 14px;
    margin-bottom: 10px;
}

/* ---------------- FOOTER ---------------- */

.footer {
    text-align: center;
    padding: 40px 0 10px;
    color: #68776c;
    font-size: 13px;
}

.footer strong {
    color: #245b41;
}

/* ---------------- RESPONSIVE ---------------- */

@media(max-width: 768px) {

    .hero h1 {
        font-size: 38px;
    }

    .hero {
        padding-top: 35px;
    }

    .navbar {
        padding: 15px;
    }

    .brand {
        font-size: 21px;
    }

    .nav-text {
        font-size: 10px;
    }

    .result-price {
        font-size: 34px;
    }

}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Section headings */
.section-heading {
    color: #174D38 !important;
    font-size: 23px !important;
    font-weight: 800 !important;
    margin-bottom: 22px;
    padding-bottom: 10px;
    border-bottom: 2px solid #E5EDE7;
    letter-spacing: 0.3px;
}

/* Remove unwanted heading anchor icons */
.section-heading a {
    display: none !important;
}

/* Improve input labels */
.stNumberInput label {
    color: #263D32 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# 4. NAVIGATION BAR
# ==========================================================

st.html("""
<div class="navbar">

    <div class="brand">
        ESTATE<span>AI</span> 🏡
    </div>

    <div class="nav-text">
        SMART PROPERTY VALUATION
    </div>

</div>
""")


# ==========================================================
# 5. HERO SECTION
# ==========================================================

st.html("""
<div class="hero">

    <div class="hero-badge">
        ✦ ARTIFICIAL INTELLIGENCE POWERED
    </div>

    <h1>
        Find the Value of<br>
        Your <span>Dream Home.</span>
    </h1>

    <p>
        Discover an estimated property value with our
        Artificial Neural Network. Enter the housing
        information below and get an instant AI-powered
        prediction.
    </p>

</div>
""")


# ==========================================================
# 6. INFORMATION CARDS
# ==========================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.html("""
    <div class="stat-card">
        <div class="stat-number">ANN</div>
        <div class="stat-label">Deep Learning Technology</div>
    </div>
    """)

with c2:
    st.html("""
    <div class="stat-card">
        <div class="stat-number">08</div>
        <div class="stat-label">Housing Features</div>
    </div>
    """)

with c3:
    st.html("""
    <div class="stat-card">
        <div class="stat-number">AI</div>
        <div class="stat-label">Intelligent Prediction</div>
    </div>
    """)


st.write("")
st.write("")


# ==========================================================
# 7. PROPERTY FORM HEADER
# ==========================================================

st.html("""
<style>

.section-header {
    background: linear-gradient(135deg, #FFFFFF, #F5F9F6);
    padding: 25px 30px;
    border: 1px solid #DCE8DF;
    border-left: 6px solid #216747;
    border-radius: 14px;
    margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(25, 75, 50, 0.06);
}

.section-header h2 {
    color: #174D38 !important;
    font-size: 25px !important;
    font-weight: 800 !important;
    margin: 0 0 10px 0 !important;
    letter-spacing: 0.3px;
}

.section-header p {
    color: #64756A !important;
    font-size: 14px !important;
    margin: 0 !important;
    line-height: 1.6;
}

</style>

<div class="section-header">

    <h2>🏠 Property Details</h2>

    <p>
        Complete the information below to calculate
        the estimated value of your property using
        our Artificial Neural Network.
    </p>

</div>
""")

# ==========================================================
# 8. PROPERTY INPUT FORM
# ==========================================================

if artifacts_loaded:

    # Custom styling for input section headings
    st.markdown("""
    <style>

    .section-heading {
        color: #174D38 !important;
        font-size: 21px !important;
        font-weight: 800 !important;
        padding-bottom: 12px;
        margin-bottom: 20px;
        border-bottom: 2px solid #DCE9DF;
        letter-spacing: 0.3px;
    }

    </style>
    """, unsafe_allow_html=True)

    with st.form("property_prediction_form"):

        col1, col2 = st.columns(2, gap="large")

        # ==============================================
        # LEFT SIDE - LOCATION & INCOME
        # ==============================================

        with col1:

            st.markdown(
                '<div class="section-heading">📍 Location & Income</div>',
                unsafe_allow_html=True
            )

            MedInc = st.number_input(
                "Median Income",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.1,
                help="Median income of the area, measured in tens of thousands of dollars."
            )

            HouseAge = st.number_input(
                "House Age (Years)",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Median age of houses in the area."
            )

            Latitude = st.number_input(
                "Latitude",
                min_value=32.0,
                max_value=42.0,
                value=34.05,
                step=0.01,
                format="%.4f",
                help="Geographical latitude of the property area in California."
            )

            Longitude = st.number_input(
                "Longitude",
                min_value=-125.0,
                max_value=-114.0,
                value=-118.24,
                step=0.01,
                format="%.4f",
                help="Geographical longitude of the property area in California."
            )

        # ==============================================
        # RIGHT SIDE - PROPERTY CHARACTERISTICS
        # ==============================================

        with col2:

            st.markdown(
                '<div class="section-heading">🏡 Property Characteristics</div>',
                unsafe_allow_html=True
            )

            AveRooms = st.number_input(
                "Average Rooms",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.1,
                help="Average number of rooms per household."
            )

            AveBedrms = st.number_input(
                "Average Bedrooms",
                min_value=0.0,
                max_value=25.0,
                value=1.0,
                step=0.1,
                help="Average number of bedrooms per household."
            )

            Population = st.number_input(
                "Area Population",
                min_value=1.0,
                max_value=50000.0,
                value=1000.0,
                step=100.0,
                help="Total population of the area."
            )

            AveOccup = st.number_input(
                "Average Occupancy",
                min_value=0.0,
                max_value=20.0,
                value=3.0,
                step=0.1,
                help="Average number of occupants per household."
            )

        # ==============================================
        # PREDICTION BUTTON
        # ==============================================

        st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "✦ GENERATE HOUSE PRICE PREDICTION",
            use_container_width=False
        )


    # ======================================================
    # 9. PREDICTION LOGIC
    # ======================================================

    if submitted:

        input_data = pd.DataFrame([{

            "MedInc": MedInc,
            "HouseAge": HouseAge,
            "AveRooms": AveRooms,
            "AveBedrms": AveBedrms,
            "Population": Population,
            "AveOccup": AveOccup,
            "Latitude": Latitude,
            "Longitude": Longitude

        }])

        try:

            # Match training feature order

            if hasattr(scaler, "feature_names_in_"):

                input_data = input_data[
                    list(scaler.feature_names_in_)
                ]

            # Standardize input

            input_scaled = scaler.transform(input_data)

            # Predict using ANN

            prediction = model.predict(
                input_scaled,
                verbose=0
            )

            predicted_value = float(prediction[0][0])

            # Convert to USD

            predicted_usd = predicted_value * 100000

            if not np.isfinite(predicted_usd) or predicted_usd <= 0:

                st.warning(
                    "The model returned an invalid estimate. "
                    "Please check your input values."
                )

            else:

                # Display prediction

                st.html(f"""
                <div class="result-card">

                    <div class="result-tag">
                        ✦ YOUR ESTIMATED PROPERTY VALUE
                    </div>

                    <div class="result-price">
                        ${predicted_usd:,.2f}
                    </div>

                    <div class="result-note">
                        Predicted California Housing Value
                    </div>

                    <div class="result-note">
                        Model output: {predicted_value:.4f}
                    </div>

                </div>
                """)

                st.success(
                    "Your house price prediction has been generated!"
                )

                # Property input summary

                st.markdown("")

                st.html("""
                <div class="detail-card">
                    <div class="detail-title">
                        📋 Your Property Information
                    </div>
                </div>
                """)

                summary1, summary2 = st.columns(2)

                with summary1:

                    st.write(f"**Median Income:** {MedInc}")
                    st.write(f"**House Age:** {HouseAge} years")
                    st.write(f"**Average Rooms:** {AveRooms}")
                    st.write(f"**Average Bedrooms:** {AveBedrms}")

                with summary2:

                    st.write(f"**Population:** {Population}")
                    st.write(f"**Average Occupancy:** {AveOccup}")
                    st.write(f"**Latitude:** {Latitude}")
                    st.write(f"**Longitude:** {Longitude}")

                # Download prediction report

                report = pd.DataFrame([{
                    **input_data.iloc[0].to_dict(),
                    "Predicted_Value_100000_USD": predicted_value,
                    "Estimated_Price_USD": predicted_usd
                }])

                csv = report.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇ Download Prediction Report",
                    data=csv,
                    file_name="house_price_prediction.csv",
                    mime="text/csv"
                )

                st.info(
                    "The prediction is based on the California Housing "
                    "dataset and is an educational model estimate, "
                    "not an official property valuation."
                )

        except Exception as error:

            st.error(f"Prediction error: {error}")


else:

    st.warning(
        "Model files are missing. Please check that "
        "house_price_model.h5 and scaler.pkl are in the project folder."
    )


# ==========================================================
# 10. FOOTER
# ==========================================================

st.html("""
<div class="footer">

    <hr style="border-color:#deded2;">

    <p>
        <strong>ESTATEAI</strong> | SMART HOUSE PRICE PREDICTION
    </p>

    <p>
        Developed using Python • TensorFlow • Keras • Scikit-learn • Streamlit
    </p>

    <p>
        Powered by Artificial Neural Networks
    </p>

</div>
""")

