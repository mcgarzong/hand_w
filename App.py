import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os

def predictDrawing(image):
    try:
        model = tf.keras.models.load_model("model/handwritten.h5")
    except Exception as e:
        st.error("❌ No se pudo cargar el modelo. Verifica la ruta 'model/handwritten.h5'.")
        st.stop()

    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

st.set_page_config(page_title='🧠 Lienzo del Pensamiento', layout='wide')

st.title('🎨 Lienzo del Pensamiento')
st.subheader("Dibuja libremente y deja que la IA interprete tu trazo")

st.write(
    "Cada línea y forma que dibujes será procesada por una red neuronal entrenada para reconocer patrones. "
    "Puede identificar números, símbolos o cualquier figura que se parezca a los datos con los que fue entrenada. "
    "¡Explora cómo la máquina percibe tu arte!"
)

drawing_mode = "freedraw"
stroke_width = st.slider('✏️ Selecciona el grosor del trazo', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

if st.button('🔮 Interpretar dibujo'):
    if canvas_result.image_data is not None:
        os.makedirs("prediction", exist_ok=True)
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        img_path = 'prediction/img.png'
        input_image.save(img_path)
        img = Image.open(img_path)
        res = predictDrawing(img)
        st.success(f"✨ La IA interpreta que tu dibujo representa un **{res}**")
    else:
        st.warning('Por favor, dibuja algo en el lienzo antes de interpretar.')

st.sidebar.title("🧩 Acerca del experimento:")
st.sidebar.write(
    "Esta aplicación es parte de un ejercicio creativo que combina arte e inteligencia artificial. "
    "Aquí, una red neuronal analiza tus trazos y trata de interpretarlos según los patrones que ha aprendido. "
    "No siempre acertará... pero ahí está la magia de la exploración visual. 🌀"
)

st.sidebar.markdown("---")
st.sidebar.caption("Basado en un desarrollo de Vinay Uniyal, adaptado para el proyecto *Lienzo del Pensamiento*.")
