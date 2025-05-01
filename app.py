import streamlit as st
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import imageio.v2 as imageio

MEAN_PIXEL = np.array([123.68, 116.779, 103.939], dtype=np.float32)
WEIGHTS_INIT_STDEV = 0.1

def get_img(path, img_size=(512, 512)):
    img = imageio.imread(path, pilmode='RGB')
    if img.ndim != 3 or img.shape[2] != 3:
        img = np.dstack([img] * 3)
    img = np.array(Image.fromarray(img).resize(img_size))
    return img.astype(np.float32)

def save_img(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    imageio.imwrite(path, img)

def transform_net(image):
    conv1 = _conv_layer(image, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid = conv3
    for _ in range(5):
        resid = _residual_block(resid, 3)
    conv_t1 = _conv_transpose_layer(resid, 64, 3, 2)
    conv_t2 = _conv_transpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    return tf.nn.tanh(conv_t3) * 150 + 255. / 2

def _conv_layer(net, num_filters, filter_size, stride, relu=True):
    weights = _init_weights(net, num_filters, filter_size)
    net = tf.nn.conv2d(net, weights, strides=[1, stride, stride, 1], padding='SAME')
    net = _instance_norm(net)
    return tf.nn.relu(net) if relu else net

def _conv_transpose_layer(net, num_filters, filter_size, stride):
    weights = _init_weights(net, num_filters, filter_size, transpose=True)
    batch_size, rows, cols, _ = net.shape
    new_shape = tf.stack([tf.shape(net)[0], rows * stride, cols * stride, num_filters])
    net = tf.nn.conv2d_transpose(net, weights, new_shape, strides=[1, stride, stride, 1], padding='SAME')
    net = _instance_norm(net)
    return tf.nn.relu(net)

def _residual_block(net, filter_size):
    tmp = _conv_layer(net, 128, filter_size, 1)
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)

def _instance_norm(net):
    mu, sigma_sq = tf.nn.moments(net, axes=[1, 2], keepdims=True)
    shift = tf.Variable(tf.zeros(net.shape[-1]), trainable=False)
    scale = tf.Variable(tf.ones(net.shape[-1]), trainable=False)
    epsilon = 1e-3
    normalized = (net - mu) / tf.sqrt(sigma_sq + epsilon)
    return scale * normalized + shift

def _init_weights(net, out_channels, filter_size, transpose=False):
    _, _, _, in_channels = net.shape
    shape = [filter_size, filter_size, out_channels, in_channels] if transpose else [filter_size, filter_size, in_channels, out_channels]
    return tf.Variable(tf.random.truncated_normal(shape, stddev=WEIGHTS_INIT_STDEV), dtype=tf.float32)

def apply_transform(img, checkpoint_path):
    img_np = np.array(img.resize((512, 512))).astype(np.float32)
    batch = np.expand_dims(img_np, 0)

    g = tf.Graph()
    with tf.compat.v1.Session(graph=g) as sess:
        input_ph = tf.compat.v1.placeholder(tf.float32, shape=(1,) + img_np.shape, name='input_img')
        preds = transform_net(input_ph)

        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, checkpoint_path)

        out = sess.run(preds, feed_dict={input_ph: batch})
        return Image.fromarray(np.clip(out[0], 0, 255).astype(np.uint8))

# ───────────────────────────────────────────────
# Streamlit UI
# ───────────────────────────────────────────────
st.set_page_config(page_title="Style Transfer App", layout="wide")
st.title("Artistic Style Transfer")
st.markdown("Upload an image and choose a trained model to apply a unique artistic style.")

# Sidebar with inputs
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

    models_dir = "models"
    available_models = [f.replace(".ckpt", "") for f in os.listdir(models_dir) if f.endswith(".ckpt")]
    selected_model = st.selectbox("🎨 Select a style", available_models)

    generate = st.button("🔄 Generate stylized image")

# Notify if no model found
if not available_models:
    st.warning("⚠️ No trained models found in the `models/` directory.\nTrain one using:\n```bash\npython main.py --mode train\n```")

# Display columns if input is valid
if uploaded_file and selected_model:
    col1, col2 = st.columns(2)
    input_img = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.subheader("Original Image")
        st.image(input_img, use_container_width=True)

    if generate:
        model_path = os.path.join(models_dir, selected_model + ".ckpt")
        output_img = apply_transform(input_img, model_path)

        with col2:
            st.subheader(f"Stylized Image ({selected_model})")
            st.image(output_img, use_container_width=True)

            output_path = f"output_{selected_model}.jpeg"
            output_img.save(output_path)
            with open(output_path, "rb") as f:
                st.download_button("📥 Download Stylized Image", f, file_name=output_path, mime="image/jpeg")
