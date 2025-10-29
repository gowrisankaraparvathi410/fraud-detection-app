import streamlit as st
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import easyocr
from deepface import DeepFace
from io import BytesIO
from pdf2image import convert_from_bytes

# --- THEME, HEADER, SIDEBAR COLORS ---
st.set_page_config(
    page_title="Next-Gen Banking Fraud Guard",
    page_icon="🏦",
    layout="centered"
)
st.markdown("""
    <style>
    /* Sidebar styling: light background, blue text */
    [data-testid="stSidebar"] {
        background-color: #f6f8fa !important;
    }
    [data-testid="stSidebar"] * {
        color: #19529a !important;
    }
    [data-testid="stSidebar"] .css-1oe5cao,
    [data-testid="stSidebar"] .css-1v3fvcr {
        color: #134074 !important;
        font-weight: 600;
    }
    .stApp {background-color: #f8fbff;}
    .st-bh {color: #074478;}
    .stButton>button {background:#1c3977;color:white;border-radius:8px;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏦 Next-Gen Banking Fraud Guard")
st.caption("Digital Document, KYC & Transaction Validation System")

tab_labels = [
    "Doc Forgery", "Signature Check", "Aadhaar", "PAN", "KYC FaceMatch", "Unusual Txns"
]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_labels)

reader = easyocr.Reader(['en'], gpu=False)

# ====== DOCUMENT FORGERY MODULE ======
with tab1:
    st.header("Document Forgery Checker")
    ft = st.radio("Document type:", ["Image","PDF"], horizontal=True)
    colA, colB = st.columns(2)
    with colA:
        origin = st.file_uploader("Original",type=['png','jpg','jpeg','pdf'],key='origdoc')
    with colB:
        test = st.file_uploader("To Verify",type=['png','jpg','jpeg','pdf'],key='testdoc')

    def to_image(file,ftype):
        if ftype=="PDF":
            pages = convert_from_bytes(file.read())
            return np.array(pages[0])
        else:
            return cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if origin and test:
        img1 = to_image(origin,ft) if ft=="PDF" else cv2.imdecode(np.frombuffer(origin.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = to_image(test,ft) if ft=="PDF" else cv2.imdecode(np.frombuffer(test.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        g1, g2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, diff = ssim(g1,g2, full=True)
        st.write(f"*Structural similarity:* {score:.3f}")
        st.image((diff*255).astype(np.uint8), caption="Difference Map")
        st.warning("Significant difference detected!" if score < 0.87 else "No significant difference.")

# ====== SIGNATURE MATCHING ======
with tab2:
    st.header("Signature Verification")
    sigA = st.file_uploader("Reference Signature",type=['png','jpg','jpeg'],key="sig1")
    sigB = st.file_uploader("To Verify",type=['png','jpg','jpeg'],key="sig2")
    if sigA and sigB:
        imgA = cv2.imdecode(np.frombuffer(sigA.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        imgB = cv2.imdecode(np.frombuffer(sigB.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(imgA, None)
        kp2, des2 = orb.detectAndCompute(imgB, None)
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            st.write(f"Feature matches: {len(matches)}")
            st.success("Matched (Likely True Signature)" if len(matches) > 45 else "Likely Forged")
        else:
            st.warning("Cannot extract enough features.")

# ====== AADHAAR MODULE ======
with tab3:
    st.header("Aadhaar Verifier")
    anu = st.text_input("Aadhaar Number (XXXX-XXXX-XXXX)")
    if st.button("Check Aadhaar", key="aadhaarcheck"):
        valid = (len(anu) == 14 and all(x.isdigit() or x=='-' for x in anu.replace("-","")))
        st.success("Format valid." if valid else "Wrong format.")

# ====== PAN MODULE ======
with tab4:
    st.header("PAN Validator")
    pnu = st.text_input("PAN Number (ABCDE1234F)")
    if st.button("Check PAN",key="pancheck"):
        valid = (len(pnu)==10 and pnu[:5].isalpha() and pnu[5:9].isdigit() and pnu[-1].isalpha())
        st.success("Format valid." if valid else "Wrong format.")

# ====== AI KYC FACEMATCH ======
with tab5:
    st.header("AI-based KYC Face Verification")
    colx, coly = st.columns(2)
    photo = colx.file_uploader("Upload Selfie",type=['png','jpg','jpeg'],key="kycself")
    idpic = coly.file_uploader("Upload ID Headshot",type=['png','jpg','jpeg'],key="kycid")
    if photo and idpic:
        try:
            res = DeepFace.verify(np.array(Image.open(photo)), np.array(Image.open(idpic)), enforce_detection=False)
            st.write(f"Distance: {res['distance']:.3f}")
            st.success("Faces Match!" if res["verified"] else "Face mismatch detected.")
        except Exception as ex:
            st.error(f"Facial verification failed: {ex}")

# ====== ANOMALOUS TRANSACTION DETECTION ======
with tab6:
    st.header("Analyze Transaction Unusual Patterns")
    tfile = st.file_uploader("Upload Transaction CSV", type="csv")
    if tfile:
        df = pd.read_csv(tfile)
        st.dataframe(df.head(10))
        zscores = ((df.select_dtypes('number')-df.select_dtypes('number').mean())/df.select_dtypes('number').std()).abs()
        anomalies = df[zscores>3].dropna(how='all')
        st.subheader("Rows with High Z-score")
        st.dataframe(anomalies)

st.divider()
if st.button("Download Current Report as PDF"):
    st.info("Report module in beta. (Demo—extend as needed!)")
