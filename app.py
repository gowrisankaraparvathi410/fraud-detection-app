
# app.py
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import easyocr
from deepface import DeepFace
from io import BytesIO

# Optional PDF support
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.warning("PDF support not available. Install pdf2image and poppler-utils.")

# --- THEME & HEADER ---
st.set_page_config(
    page_title="Next-Gen Banking Fraud Guard",
    page_icon="🏦",
    layout="centered"
)
st.markdown("""
    <style>
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

# Initialize EasyOCR reader
try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    st.error(f"Failed to initialize EasyOCR: {e}")
    reader = None

# ====== DOCUMENT FORGERY MODULE ======
with tab1:
    st.header("Document Forgery Checker")
    ft = st.radio("Document type:", ["Image","PDF"], horizontal=True)
    
    if ft == "PDF" and not PDF_SUPPORT:
        st.error("PDF support requires pdf2image library and poppler-utils")
    
    colA, colB = st.columns(2)
    with colA:
        origin = st.file_uploader("Original",type=['png','jpg','jpeg','pdf'],key='origdoc')
    with colB:
        test = st.file_uploader("To Verify",type=['png','jpg','jpeg','pdf'],key='testdoc')

    def to_image(file, ftype):
        try:
            if ftype == "PDF":
                if not PDF_SUPPORT:
                    return None
                pages = convert_from_bytes(file.read())
                return np.array(pages[0])
            else:
                return cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None

    if origin and test:
        try:
            img1 = to_image(origin, ft)
            img2 = to_image(test, ft)
            
            if img1 is None or img2 is None:
                st.error("Failed to load images")
            else:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                g1, g2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                score, diff = ssim(g1, g2, full=True)
                st.write(f"*Structural similarity:* {score:.3f}")
                st.image((diff*255).astype(np.uint8), caption="Difference Map")
                
                if score < 0.87:
                    st.error("⚠ Significant difference detected - Possible forgery!")
                else:
                    st.success("✓ No significant difference - Documents match")
        except Exception as e:
            st.error(f"Analysis failed: {e}")

# ====== SIGNATURE MATCHING ======
with tab2:
    st.header("Signature Verification")
    sigA = st.file_uploader("Reference Signature",type=['png','jpg','jpeg'],key="sig1")
    sigB = st.file_uploader("To Verify",type=['png','jpg','jpeg'],key="sig2")
    
    if sigA and sigB:
        try:
            imgA = cv2.imdecode(np.frombuffer(sigA.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            imgB = cv2.imdecode(np.frombuffer(sigB.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(imgA, None)
            kp2, des2 = orb.detectAndCompute(imgB, None)
            
            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                st.write(f"*Feature matches:* {len(matches)}")
                
                if len(matches) > 45:
                    st.success("✓ Signatures Match - Likely Authentic")
                else:
                    st.error("⚠ Signatures Don't Match - Possible Forgery")
            else:
                st.warning("Cannot extract enough features from signatures.")
        except Exception as e:
            st.error(f"Signature verification failed: {e}")

# ====== AADHAAR MODULE ======
with tab3:
    st.header("Aadhaar Verifier")
    anu = st.text_input("Aadhaar Number (XXXX-XXXX-XXXX)")
    
    if st.button("Check Aadhaar", key="aadhaarcheck"):
        try:
            cleaned = anu.replace("-", "").replace(" ", "")
            valid = len(cleaned) == 12 and cleaned.isdigit()
            
            if valid:
                st.success("✓ Aadhaar format is valid")
            else:
                st.error("⚠ Invalid Aadhaar format. Must be 12 digits.")
        except Exception as e:
            st.error(f"Validation failed: {e}")

# ====== PAN MODULE ======
with tab4:
    st.header("PAN Validator")
    pnu = st.text_input("PAN Number (ABCDE1234F)").upper()
    
    if st.button("Check PAN", key="pancheck"):
        try:
            valid = (len(pnu) == 10 and 
                    pnu[:5].isalpha() and 
                    pnu[5:9].isdigit() and 
                    pnu[-1].isalpha())
            
            if valid:
                st.success("✓ PAN format is valid")
            else:
                st.error("⚠ Invalid PAN format. Format: ABCDE1234F (5 letters, 4 digits, 1 letter)")
        except Exception as e:
            st.error(f"Validation failed: {e}")

# ====== AI KYC FACEMATCH ======
with tab5:
    st.header("AI-based KYC Face Verification")
    colx, coly = st.columns(2)
    photo = colx.file_uploader("Upload Selfie",type=['png','jpg','jpeg'],key="kycself")
    idpic = coly.file_uploader("Upload ID Headshot",type=['png','jpg','jpeg'],key="kycid")
    
    if photo and idpic:
        try:
            with st.spinner("Analyzing faces..."):
                photo_img = np.array(Image.open(photo))
                id_img = np.array(Image.open(idpic))
                
                res = DeepFace.verify(
                    photo_img, 
                    id_img, 
                    enforce_detection=False,
                    model_name="VGG-Face"
                )
                
                st.write(f"*Face Distance:* {res['distance']:.3f}")
                
                if res["verified"]:
                    st.success("✓ Faces Match - Identity Verified!")
                else:
                    st.error("⚠ Face Mismatch Detected - Verification Failed")
        except Exception as ex:
            st.error(f"Facial verification failed: {ex}")
            st.info("Try using clearer images with visible faces")

# ====== ANOMALOUS TRANSACTION DETECTION ======
with tab6:
    st.header("Analyze Transaction Unusual Patterns")
    tfile = st.file_uploader("Upload Transaction CSV", type="csv")
    
    if tfile:
        try:
            df = pd.read_csv(tfile)
            st.write("*Data Preview:*")
            st.dataframe(df.head(10))
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                # Calculate Z-scores
                zscores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
                
                # Find anomalies (Z-score > 3)
                anomaly_mask = (zscores > 3).any(axis=1)
                anomalies = df[anomaly_mask]
                
                st.subheader("Anomalous Transactions Detected")
                st.write(f"*Found {len(anomalies)} unusual transaction(s)*")
                
                if len(anomalies) > 0:
                    st.dataframe(anomalies)
                    st.warning("⚠ These transactions show statistical anomalies (Z-score > 3)")
                else:
                    st.success("✓ No anomalies detected in the dataset")
            else:
                st.warning("No numeric columns found for analysis")
        except Exception as e:
            st.error(f"Transaction analysis failed: {e}")

# FINAL NOTE
st.divider()
if st.button("Download Current Report as PDF"):
    st.info("📄 Report generation feature coming soon!")
    
