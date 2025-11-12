import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

# Page setup
st.set_page_config(page_title="Rolling Defects Detection", layout="centered")

# --- 1 Model Path ---
MODEL_PATH = "best.pt"  # Ensure your trained model file is in the same folder

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

# --- 2 Defect Class Names and Detailed Descriptions ---
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']
DESC = {
    "crazing": """**Crazing:**  
Crazing refers to fine, shallow surface cracks that appear on steel due to uneven cooling or thermal stresses, typically occurring when hot steel (around 800–1200°C) is cooled too rapidly.  
These cracks form a network-like pattern on the surface and can weaken the material’s appearance and strength.  
Crazing usually results from sudden quenching, irregular furnace heating, or poor descaling that causes localized stress points.  
It can be reduced by maintaining uniform heating, ensuring gradual and controlled cooling, using even descaling methods, and improving furnace temperature control.""",

    "inclusion": """**Inclusion:**  
Inclusions are non-metallic particles such as oxides, sulfides, or silicates that become trapped within the steel during melting or casting, usually at temperatures above 1500°C.  
They appear as dark or irregular spots and negatively affect the steel’s surface finish and mechanical properties.  
Common causes include impurities in raw materials, poor refining, or slag carryover.  
To reduce inclusions, steelmaking should involve vacuum degassing, argon purging, clean ladles, and effective desulfurization and deoxidation to ensure high purity before casting.""",

    "patches": """**Patches:**  
Patches are uneven or discolored regions on the steel surface caused by oxidation, irregular descaling, or improper cleaning during hot rolling, generally at 900–1200°C.  
These defects arise from non-uniform furnace heating, scale accumulation, or uneven pressure from descaling jets.  
They can be prevented by maintaining uniform furnace temperatures, optimizing descaling water pressure and distribution, and reducing delays between reheating and rolling.""",

    "pitted_surface": """**Pitted Surface:**  
A pitted surface shows small depressions or holes formed by localized oxidation, inclusion pull-out, or gas entrapment during rolling or cooling.  
This typically occurs during hot rolling (900–1200°C) or pickling.  
Pitting can result from moisture on the surface, trapped gases, or poor cleaning before rolling.  
It can be reduced by ensuring dry and clean surfaces, proper degassing, effective descaling, and avoiding exposure to moisture or humid air when the steel is hot.""",

    "rolled_in_scale": """**Rolled-in Scale:**  
Rolled-in scale appears as dark streaks or rough patches when oxide layers formed at high temperatures (900–1100°C) are pressed into the steel surface during rolling.  
It occurs due to inadequate descaling, overheating, or residual scale on rollers.  
Preventive measures include using high-pressure water descaling before each rolling pass, keeping rolls and guides clean, minimizing furnace-to-roll time, and maintaining low oxygen levels in reheating furnaces.""",

    "scratches": """**Scratches:**  
Scratches are linear grooves or marks caused by mechanical abrasion or contact with rough surfaces during handling, rolling, or coiling.  
They can occur at both high and ambient temperatures.  
The main causes include dirty or rough rollers, debris on conveyors, or misalignment of mill guides.  
To reduce scratches, equipment surfaces should be kept clean and smooth, proper alignment should be maintained, and non-abrasive handling tools or protective coatings should be used during and after rolling."""
}

def severity(xyxy, shape):
    """Estimate defect severity based on area ratio."""
    x1, y1, x2, y2 = xyxy
    ratio = ((x2 - x1) * (y2 - y1)) / (shape[0] * shape[1])
    return "Mild" if ratio < 0.005 else "Moderate" if ratio < 0.02 else "Severe"

# --- 3 App Header ---
st.title("Rolling Defects Detection System")
st.subheader("Made by Shvojas Aditya Reg No: 23BME1195 , Suyash 22BCE1437")
st.write("Upload a steel surface image to automatically detect and describe rolling defects using a YOLOv8 AI-based system.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded:
    # --- 4 Preprocessing: Brightness & Contrast Enhancement ---
    img = Image.open(uploaded).convert("RGB")
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    img_cv = cv2.convertScaleAbs(img_cv, alpha=1.2, beta=15)  # slight contrast & brightness boost
    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    st.image(img, caption="Uploaded Image (Enhanced)", use_column_width=True)

    # --- 5 YOLO Inference ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        res = model.predict(source=tmp.name, conf=0.35, imgsz=768, verbose=False)[0]

    # --- 6 Annotate Results ---
    try:
        annotated = res.plot(line_width=1, font_size=0.5)
    except TypeError:
        annotated = res.plot()

    # --- 7 Filter Out Tiny Boxes ---
    boxes = getattr(res, "boxes", [])
    filtered_boxes = []
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
        area = (x2 - x1) * (y2 - y1)
        if area > 400:  # ignore small detections
            filtered_boxes.append(b)
    boxes = filtered_boxes

    # --- 8 Show Annotated Image ---
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Detected Defects (Refined)", use_column_width=True)

    # --- 9 Display Defect Results ---
    if len(boxes) == 0:
        st.info("No significant rolling defects detected.")
    else:
        st.subheader("Detected Defects")
        for b in boxes:
            cid, conf = int(b.cls), float(b.conf)
            name = CLASSES[cid] if cid < len(CLASSES) else f"class_{cid}"
            sev = severity([int(x) for x in b.xyxy[0]], img.size[::-1])
            st.markdown(f"**{name.capitalize()}** — Confidence: `{conf:.2f}` — Severity: `{sev}`")
            st.write(DESC.get(name, "No description available."))

        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        st.download_button(
            label="Download Annotated Image",
            data=buffer.tobytes(),
            file_name="rolling_defects_detected.jpg",
            mime="image/jpeg"
        )

