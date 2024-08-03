import streamlit as st
import numpy as np
import pandas as pd
import librosa

import plotly.express as px

from streamlit_vertical_slider import vertical_slider

st.title("Equalizador")
st.audio(data="audio_samples/song_of_the_sunbird.wav", format="audio/wav")

filters = [
    "32Hz",
    "64Hz",
    "128Hz",
    "256Hz",
    "512Hz",
    "1KHz",
    "2KHz",
    "4KHz",
    "8KHz",
    "16KHz",
]

cols = st.columns(10)

for filter, col in zip(filters, cols):
    with col:
        val = vertical_slider(
            label=filter,
            key=filter,
            height=100,
            thumb_shape="square",
            step=1,
            default_value=0,
            min_value=-10,
            max_value=10,
            track_color="blue",
            slider_color="lighgray",
            thumb_color="orange",
            value_always_visible=False
        )

array, sampling_rate = librosa.load(librosa.ex("trumpet"))

S = librosa.stft(array, n_fft=512 * 16, window="hann")
S_db = librosa.amplitude_to_db(np.abs(S * S), ref=0.0, top_db=120)
freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=512 * 16)
spectrum = np.mean(S_db, axis=1)
data = pd.Series(spectrum, index=freqs)

fig = px.line(data)
st.plotly_chart(fig)

