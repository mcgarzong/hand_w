import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32')
    img = img / 255
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

st.set_page_config(page_title='🔮 Oráculo Numérico', layout='wide')

st.title("🔮 El Oráculo Numérico")
st.markdown("""
El **Oráculo Numérico** traduce los símbolos que trazas con tu mano en energía digital.  
Cada número contiene una vibración, un significado, una huella que la inteligencia interpreta.  
Dibuja un número en el lienzo y deja que el oráculo revele su identidad.
""")

st.subheader("✍️ Dibuja tu número en el panel y presiona **“Revelar número”**")

stroke_width = st.slider('🖌️ Ajusta el trazo del símbolo', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.0)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

if st.button('🔍 Revelar número'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.success(f"✨ El número revelado por el oráculo es: **{res}**")
    else:
        st.warning("⚠️ Traza un símbolo antes de invocar al oráculo.")

st.sidebar.title("📜 Sobre el Oráculo")
st.sidebar.markdown("""
El **Oráculo Numérico** es un sistema de visión artificial  
entrenado para **reconocer dígitos escritos a mano**.  

Funciona mediante una **red neuronal artificial (RNA)**  
que traduce los trazos humanos al lenguaje de los datos.  

Creado como parte de una exploración entre lo **humano y lo digital**.  
""")
