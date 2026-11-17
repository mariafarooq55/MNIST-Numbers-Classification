
import numpy as np
import joblib
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", page_icon="✏️")


# 1. Load the trained model only once (cache = remember it, do not reload every click)

def load_model():
    return joblib.load("mnist_model.pkl")

model = load_model()


# 2. Turn any picture of a digit into the same format as MNIST:
#    28x28 pixels, white digit on black, centred, values from 0 to 1
def prepare_digit(gray):
    gray = np.asarray(gray, dtype=np.float32)
    if gray.max() < 30:                                    # nothing drawn
        return None
    rows = np.where(gray.max(axis=1) > 30)[0]
    cols = np.where(gray.max(axis=0) > 30)[0]
    digit = gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]          # crop around the digit
    h, w = digit.shape
    scale = 20.0 / max(h, w)                                           # longest side becomes 20 pixels
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = np.array(Image.fromarray(digit.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS), dtype=np.float32)
    box = np.zeros((28, 28), dtype=np.float32)                         # place it in a 28x28 black box
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    box[top:top + new_h, left:left + new_w] = digit
    total = box.sum()                                                  # move the centre of the ink to the middle
    if total > 0:
        r, c = np.indices(box.shape)
        dy = int(round(14 - (r * box).sum() / total))
        dx = int(round(14 - (c * box).sum() / total))
        box = np.roll(np.roll(box, dy, axis=0), dx, axis=1)
    return (box / 255.0).reshape(1, 784)                               # scale to 0-1 and flatten


# 3. Page layout
st.title("✏️ Handwritten Digit Recognizer")
st.write("Draw a digit from **0 to 9**, or upload a photo of one. The model trained on MNIST will guess it.")

tab_draw, tab_upload = st.tabs(["✏️ Draw", "📤 Upload an image"])
gray = None


with tab_draw:
    canvas_kwargs = dict(
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    try:
        canvas = st_canvas(**canvas_kwargs, return_image_data=True)
    except TypeError:
        canvas = st_canvas(**canvas_kwargs)

    st.caption("Draw big, in the middle. Use the 🗑️ icon under the box to clear it.")
    if canvas.image_data is not None:
        gray = np.array(canvas.image_data)[:, :, 0]                    # red channel = white strokes

with tab_upload:
    file = st.file_uploader("Choose a PNG or JPG image of one digit", type=["png", "jpg", "jpeg"])
    if file is not None:
        img = np.array(Image.open(file).convert("L"), dtype=np.float32)
        if img.mean() > 127:                                           # dark ink on white paper → flip colours
            img = 255 - img
        gray = img
        st.image(file, caption="Your image", width=200)

# 4. Predict
x = prepare_digit(gray) if gray is not None else None
if x is None:
    st.info("Waiting for a digit… ✍️")
else:
    probs = model.predict_proba(x)[0]
    digit = int(model.classes_[probs.argmax()])

    left, right = st.columns([1, 2])
    with left:
        st.image(x.reshape(28, 28), caption="What the model sees (28×28)", width=140, clamp=True)
    with right:
        st.metric("Prediction", digit, f"{probs.max():.0%} sure")
    st.bar_chart({"probability": {str(int(c)): float(p) for c, p in zip(model.classes_, probs)}})