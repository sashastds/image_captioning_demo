import sys
import os
import io
import gc
import warnings
import pickle
import streamlit as st
from pathlib import Path

from caption_utils import (
    download_file_from_google_drive,
    load_models,
    image_load,
    image_process,
    get_captions,
)


### GDRIVE IDS
CAPTION_MODEL_ID = "1fgnUGsS1bl_lZk_cZP-FjJ1GwS5Sr4A7"
INCEPTION_ID = "1vBILINRhOD6FhWJvY-sueiOmmD0HYrnm"
W2I_ID = "1h_C_IOPnmpLiGzgzrF1KOeS7QRe-oXAf"
I2W_ID = "1dD4HuQPDhKCRAxaPsYa3YnWFY2ZG3WgG"
SPEC_TOKENS_ID = "1nf4NNcr92QRdHqg1Lzw3AB-TEDYndfr2"

### PATHS
UTILITIES_PATH = Path("./utilities/")

CAPTION_MODEL_PATH = UTILITIES_PATH / "caption_net.pth"
INCEPTION_PATH = UTILITIES_PATH / "bh_inception.pth"
W2I_PATH = UTILITIES_PATH / "word2idx.pkl"
I2W_PATH = UTILITIES_PATH / "idx2word.pth"
SPEC_TOKENS_PATH = UTILITIES_PATH / "spec_tokens.pth"

TMP_DIR = UTILITIES_PATH / "tmp/"
TMP_IMG_PATH = TMP_DIR / "img.jpg"

new_image = False

st.set_page_config(
    page_title="Small, but awesome image captioning demo",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Image Captioning Demo\nSmall, but awesome")


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def instantiate(pseudo_inception=False):

    if not os.path.exists(UTILITIES_PATH):
        os.mkdir(UTILITIES_PATH)
        
    message_slot = st.empty()
    message_slot.info("downloading models, this might take some time..")
    for g_id, p in zip(
        [CAPTION_MODEL_ID, INCEPTION_ID, W2I_ID, I2W_ID, SPEC_TOKENS_ID], 
        [CAPTION_MODEL_PATH, INCEPTION_PATH, W2I_PATH, I2W_PATH, SPEC_TOKENS_PATH]
    ):
        if not os.path.exists(p):
            download_file_from_google_drive(g_id, p)
            

    with open(W2I_PATH, "rb") as f:
        word2idx = pickle.load(f)
    with open(I2W_PATH, "rb") as f:
        idx2word = pickle.load(f)
    with open(SPEC_TOKENS_PATH, "rb") as f:
        spec_tokens = pickle.load(f)
        
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN = (
        spec_tokens[x] for x in ["BOS_token", "EOS_token", "PAD_token", "UNK_token"]
    )
    
    BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX = (
        word2idx[spec_tokens[x]]
        for x in ["BOS_token", "EOS_token", "PAD_token", "UNK_token"]
    )
    
    VOCAB_SIZE = len(word2idx)
            
    with warnings.catch_warnings():
        message_slot.info("initializing models, please, wait a little..")
        warnings.simplefilter("ignore", FutureWarning)
        inception, caption_model, device = load_models(
            INCEPTION_PATH, CAPTION_MODEL_PATH, VOCAB_SIZE, PAD_IDX
        )
        message_slot.empty()
    exclude_from_captions = [BOS_IDX, PAD_IDX, UNK_IDX]

    return inception, caption_model, idx2word, BOS_IDX, EOS_IDX, exclude_from_captions


@st.cache(suppress_st_warning=True, ttl=3600, max_entries=1, show_spinner=False)
def load_image(img):

    global new_image
    new_image = True

    if isinstance(img, str):
        image = image_load(img)
    else:
        img_bytes = img.read()
        if not os.path.exists(TMP_DIR):
            os.mkdir(TMP_DIR)
        with open(TMP_IMG_PATH, "wb") as f:
            f.write(img_bytes)
        image = image_load(TMP_IMG_PATH)
    return image


def main():

    (
        inception,
        caption_model,
        idx2word,
        BOS_IDX,
        EOS_IDX,
        exclude_from_captions,
    ) = instantiate(pseudo_inception = False)

    spinner_slot = st.empty()
    left, right = st.beta_columns((1, 1))
    image_slot = left
    caption_slot = right

    is_loaded = False
    img_loaded = st.file_uploader(
        label="Upload your image in .jpg format", type=["jpg", "jpeg"]
    )
    load_status_slot = st.empty()

    if img_loaded:
        img = load_image(img_loaded)
        is_loaded = True
        img, img_for_net, img_size_init = image_process(img)
        image_slot.image(img, use_column_width=True, width=500)
        if new_image:
            load_status_slot.success("Image loaded!")
    MAX_CAPTION_LEN = st.sidebar.number_input(
        "Select maximal caption length:", min_value=1, max_value=20, value=8, step=1
    )
    SAMPLING = st.sidebar.radio(
        "Should sampling be done?", ("Sampling", "Use Best Option")
    )
    SAMPLE = SAMPLING == "Sampling"

    n_captions_slot = st.sidebar.empty()
    slider_slot = st.sidebar.empty()

    if SAMPLE:
        N_CAPTIONS = n_captions_slot.number_input(
            "Select number of captions generated:",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
        )
        TEMPERATURE = slider_slot.slider(
            "Set temperature for sampling:",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
        )
    else:
        N_CAPTIONS = 1
        TEMPERATURE = 1
        slider_slot.markdown(
            "No sampling would be made while generating the one and only caption variant"
        )
    button_slot = st.sidebar.empty()
    warning_slot = st.sidebar.empty()
    authors_slot = st.sidebar.empty()

    if button_slot.button("Generate captions!"):
        load_status_slot.empty()
        if is_loaded:
            spinner_slot.info("Generating...")
            generated_captions = get_captions(
                img_for_net,
                inception,
                caption_model,
                idx2word,
                exclude_from_captions,
                (BOS_IDX,),
                EOS_IDX,
                N_CAPTIONS,
                TEMPERATURE,
                SAMPLE,
                MAX_CAPTION_LEN,
            )
            spinner_slot.empty()
            caption_slot.header("Let's see what is going on there..")
            caption_slot.markdown("  \n".join(generated_captions))
        else:
            warning_slot.warning("Please, upload your image first")
    authors_slot.markdown(
      """\
      <span style="color:black;font-size:8"><p>\
      made by\
      <a style="color:mediumorchid" href="https://www.linkedin.com/in/aleksandr-nalitkin/">aleksandr</a>
      """,
      unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
