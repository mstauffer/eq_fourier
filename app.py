import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from streamlit_vertical_slider import vertical_slider
import tempfile
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import time

from frequency_manipulation.utils import create_bandpass_filter, calculate_band_magnitudes

st.set_page_config(layout='wide')

t = tempfile.TemporaryDirectory()
if 'audio_set' not in st.session_state:
    st.session_state.audio_set = False

music_library = {
    'Miles Davis - Stella by Starlight (Jazz, Trumpete)': 'audio_samples/miles_davis_stella_by_starlight.wav',
    'Boney James - Butter (Jazz, Sax)': 'audio_samples/boney_james_butter.wav',
    'Shaman - Fairy Tale (Rock, Coro Lírico no começo)': 'audio_samples/shaman_fairy_tale.wav',
    'Beatles - Twist and Shout (Rock, Guitarra)': 'audio_samples/twist_and_shout.wav',
    'Soft Machine - Song of the Sunbird (Jazz, variados timbres de teclado)': 'audio_samples/song_of_the_sunbird.wav',
    "One Republic - I Ain't Worried (Pop)": 'audio_samples/onerepublic_iaintworried.wav',
    'Rob Araujo - Deep (Jazz/Eletrônico)': 'audio_samples/rob_araujo_deep.wav',
    'Justin Bieber - Yummy (Pop)': 'audio_samples/justin_bieber_yummy.wav'
}

cs = st.columns(2)
with cs[0]:
    uploaded_file = st.file_uploader("Escolha um arquivo de áudio ou")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        with open(f'{t.name}/audio', 'wb') as f:
            f.write(bytes_data)
            st.session_state.audio_set = True
        audio_path = f'{t.name}/audio'
with cs[1]:
    music = st.selectbox('Escolha uma música da nossa lib', list(music_library.keys()))
    if music:
        audio_path = music_library.get(music)
        st.session_state.audio_set = True

if st.session_state.audio_set:
    samples, sampling_rate = librosa.load(audio_path, sr=None)

    slider_col, plot_col = st.columns(2)
    
    with slider_col:
        st.write('Original audio')
        st.audio(data=audio_path, format="audio/wav")

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
        gains = []

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
                    slider_color="lightgray",
                    thumb_color="orange",
                    value_always_visible=False
                )
                gains.append(val)

    with plot_col:
        filter_length = st.number_input('Filter Length', min_value=1, max_value=500, value=100)

    M = filter_length
    bandwidth = 0.2

    equalized_signal = np.zeros_like(samples)
    center_frequencies = [32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000]

    for center_freq, gain in zip(center_frequencies, gains):
        h = create_bandpass_filter(center_freq, bandwidth, sampling_rate, M)
        filtered_signal = np.convolve(samples, h, mode='same')
        
        # Apply gain and add to the equalized signal
        equalized_signal += filtered_signal * (10 ** (gain / 20))

    # Normalization to prevent clipping
    max_amplitude = np.max(np.abs(equalized_signal))
    if max_amplitude > 1:
        equalized_signal /= max_amplitude
    
    output_path = 'equalized_audio.wav'
    sf.write(output_path, equalized_signal, sampling_rate)

    with slider_col:
        st.write('Filtered audio')
        st.audio(data=output_path, format="audio/wav")

    with plot_col:
        graphic_col, frame_rate_col = st.columns(2)
        with graphic_col:
            if st.button("Start Visualization"):
                st.session_state.start_visualization = True
        
        with frame_rate_col:
            frame_rate_val = st.number_input('Frame Rate',
                                         min_value=15,
                                         max_value=100,
                                         value=15)
        
        chart_placeholder = st.empty()

        if 'start_visualization' in st.session_state and st.session_state.start_visualization:
            frame_rate = frame_rate_val
            chunk_size = sampling_rate // frame_rate
            audio_data, _ = sf.read(output_path)
            start_time = time.time()

            max_magnitude = 0.5

            for i in range(0, len(audio_data), chunk_size):
                elapsed_time = time.time() - start_time
                
                expected_time = i / sampling_rate

                if elapsed_time < expected_time:
                    time.sleep(expected_time - elapsed_time)

                audio_chunk = audio_data[i:i + chunk_size]

                band_magnitudes = calculate_band_magnitudes(audio_chunk, sampling_rate, center_frequencies)

                plt.figure(figsize=(8, 4))
                plt.bar(range(len(center_frequencies)), band_magnitudes, tick_label=[f"{f}Hz" for f in center_frequencies])
                plt.xlabel('Frequency Band')
                plt.ylabel('Magnitude')
                plt.ylim(0, max_magnitude)

                chart_placeholder.pyplot(plt)
                plt.close()

                time.sleep(1 / frame_rate)
