import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
import matplotlib.pyplot as plt
import os

# -------------------------------
# Model configuration (must match training)
# -------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = r'C:\Users\shubh\OneDrive\projects\Minor-Project under DevraniMam\mini-project\federated_lung_nodule_unetpp_effb3(2).pth'

@st.cache_resource
def load_model():
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image):
    """Convert PIL image to tensor for model."""
    img = np.array(image.convert('L'))
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)   # [1, H, W]
    img = np.expand_dims(img, axis=0)   # [1, C, H, W]
    return torch.tensor(img, dtype=torch.float32)

def compute_metrics(pred_mask, gt_mask):
    """Compute Dice and IoU between binary masks."""
    pred = pred_mask.flatten().astype(np.float32)
    gt = gt_mask.flatten().astype(np.float32)
    intersection = np.sum(pred * gt)
    dice = (2. * intersection + 1e-6) / (np.sum(pred) + np.sum(gt) + 1e-6)
    iou = (intersection + 1e-6) / (np.sum(pred) + np.sum(gt) - intersection + 1e-6)
    return dice, iou

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Lung Nodule Segmentation", layout="wide")
st.title("🫁 Lung Nodule Segmentation")
st.markdown("Upload a CT slice (PNG) and optional ground truth mask to detect and segment lung nodules.")

# Load model once
model = load_model()

# ---------- Step 1: Inputs ----------
st.markdown("### 1. Upload Images")
col_input1, col_input2 = st.columns(2)

with col_input1:
    uploaded_ct = st.file_uploader("Upload CT image (PNG)", type=["png", "jpg", "jpeg"], key="ct")

with col_input2:
    uploaded_mask = st.file_uploader("Upload ground truth mask (optional, PNG)", type=["png", "jpg", "jpeg"], key="mask")

# ---------- Step 2: Process and Display ----------
if uploaded_ct is not None:
    # Load and preprocess CT
    ct_image = Image.open(uploaded_ct).convert('L')
    ct_tensor = preprocess_image(ct_image).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(ct_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        pred_mask = (probs > 0.5).astype(np.uint8)

    # Resize prediction back to original image size
    original_size = ct_image.size  # (width, height)
    pred_mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)

    # Prepare ground truth if provided
    gt_binary = None
    if uploaded_mask is not None:
        gt_image = Image.open(uploaded_mask).convert('L')
        gt_np = np.array(gt_image)
        gt_resized = cv2.resize(gt_np, original_size, interpolation=cv2.INTER_NEAREST)
        gt_binary = (gt_resized > 127).astype(np.uint8)

    # Create overlay (prediction over CT)
    ct_np = np.array(ct_image)
    overlay = cv2.addWeighted(ct_np, 0.7, (pred_mask_resized * 255).astype(np.uint8), 0.3, 0)

    # ---------- Horizontal image display ----------
    st.markdown("### 2. Results")
    num_cols = 3 + (1 if gt_binary is not None else 0)
    cols = st.columns(num_cols)

    # Original CT
    with cols[0]:
        st.subheader("Original CT")
        st.image(ct_image, use_container_width=True)

    # Ground Truth (if available)
    if gt_binary is not None:
        with cols[1]:
            st.subheader("Ground Truth")
            st.image(gt_binary * 255, use_container_width=True, clamp=True)
        pred_col = 2
        overlay_col = 3
    else:
        pred_col = 1
        overlay_col = 2

    # Predicted Mask
    with cols[pred_col]:
        st.subheader("Predicted Mask")
        st.image(pred_mask_resized * 255, use_container_width=True, clamp=True)

    # Overlay
    with cols[overlay_col]:
        st.subheader("Overlay")
        st.image(overlay, use_container_width=True, clamp=True)

    # ---------- Metrics and additional visualizations ----------
    st.markdown("---")
    st.markdown("### 3. Metrics & Analysis")

    # Predicted nodule area
    nodule_area = np.sum(pred_mask_resized)
    st.metric("Predicted Nodule Area (pixels)", f"{nodule_area}")

    # Dice and IoU if GT provided
    if gt_binary is not None:
        dice, iou = compute_metrics(pred_mask_resized, gt_binary)
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Dice Coefficient", f"{dice:.4f}")
        col_m2.metric("IoU (Jaccard)", f"{iou:.4f}")
    else:
        st.info("Upload a ground truth mask to compute Dice and IoU.")

    # Probability heatmap (1/3 screen width)
    st.markdown("#### Probability Heatmap")
    col_heatmap, _ = st.columns([1, 2])  # 1/3 width for heatmap
    with col_heatmap:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(probs, cmap='viridis', vmin=0, vmax=1)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)