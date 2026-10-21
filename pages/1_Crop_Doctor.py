"""Crop Doctor page."""
import streamlit as st
from PIL import Image

from modules.crop_doctor import advice_for, load_model_and_labels, predict, pretty_name

st.set_page_config(page_title="Crop Doctor - MandiMitra AI", page_icon="🌿")
st.title("🌿 Crop Doctor")
st.caption("Upload a clear photo of a single leaf for a preliminary disease screening.")


@st.cache_resource
def get_model():
    return load_model_and_labels()


model, class_names = get_model()

st.info(
    "Supported: Potato and Tomato leaves (early blight, late blight, healthy). "
    "Trained on lab-condition PlantVillage images, so accuracy on real field photos may be lower."
)

uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", width=300)

    with st.spinner("Analysing..."):
        top3, reliable = predict(model, class_names, image)

    best_label, best_conf = top3[0]

    if reliable:
        st.success(f"Likely: {pretty_name(best_label)} ({best_conf:.1%} model confidence)")
        st.write(advice_for(best_label))
    else:
        st.warning(
            "Unable to identify reliably. Try a clearer, well-lit photo of one leaf, "
            "or consult a local agriculture expert."
        )

    st.subheader("Top 3 predictions")
    for label, conf in top3:
        st.write(f"{pretty_name(label)}: {conf:.1%}")
        st.progress(conf)

    st.caption(
        "This is a preliminary screening by an AI model, not a confirmed diagnosis. "
        "Model confidence does not guarantee correctness."
    )