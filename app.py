import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from streamlit_vertical_slider import vertical_slider
import tempfile

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
    music = st.selectbox('Escolha uma música da nossa lib', [
        'Miles Davis - Stella by Starlight (Jazz, Trumpete)',
        'Boney James - Butter (Jazz, Sax)',
        'Shaman - Fairy Tale (Rock, Coro Lírico no começo)',
        'Beatles - Twist and Shout (Rock, Guitarra)',
        'Soft Machine - Song of the Sunbird (Jazz, variados timbres de teclado)',
        "One Republic - I Ain't Worried (Pop)",
        'Rob Araujo - Deep (Jazz/Eletrônico)',
        'Justin Bieber - Yummy (Pop)'
    ])
    if music:
        audio_path = music_library.get(music)
        st.session_state.audio_set = True

if st.session_state.audio_set:
    samples, sampling_rate = librosa.load(audio_path, sr=None)

    st.title("Equalizador")
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
                slider_color="lighgray",
                thumb_color="orange",
                value_always_visible=False
            )
            gains.append(val)

    def create_bandpass_filter(center_freq, bandwidth, fs, M):
        nyquist = 0.5 * fs
        f_low = (center_freq - bandwidth / 2) / nyquist
        f_high = (center_freq + bandwidth / 2) / nyquist

        # Use sinc to create bandpass filter
        n = np.arange(-M // 2, M // 2 + 1)
        h = np.sinc(2 * f_high * n) - np.sinc(2 * f_low * n)
        window = np.hamming(M + 1)
        h *= window
        h /= np.sum(h)
        
        return h

    M = 100  # Filter length
    bandwidth = 0.2

    equalized_signal = np.zeros_like(samples)

    center_frequencies = [32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000]

    for center_freq, gain in zip(center_frequencies, gains):
        h = create_bandpass_filter(center_freq, bandwidth, sampling_rate, M)
        filtered_signal = np.convolve(samples, h, mode='same')
        
        # Apply gain and add to the equalized signal
        equalized_signal += filtered_signal * (10 ** (gain / 20))

    # normalization for prevent clipping
    max_amplitude = np.max(np.abs(equalized_signal))
    if max_amplitude > 1:
        equalized_signal /= max_amplitude

    output_path = 'equalized_audio.wav'
    sf.write(output_path, equalized_signal, sampling_rate)

    st.write('Filtered audio')
    st.audio(data=output_path, format="audio/wav")
