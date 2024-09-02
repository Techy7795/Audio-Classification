import os
import time
import argparse
import logging
import pickle
import warnings
from baby_cry_detection.pc_methods.feature_engineer import FeatureEngineer
from baby_cry_detection.pc_methods.majority_voter import MajorityVoter
from baby_cry_detection.pc_methods.baby_cry_predictor import BabyCryPredictor
import librosa
import sounddevice as sd
import soundfile as sf

def record_audio(file_path, duration, output_path, samplerate=44100, channels=2):
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    sf.write(file_path, recording, samplerate)
    print("Recording saved:", file_path)

def predict_and_play(file_path, model_path, output_path, log_path):
    # Set up logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        filename=os.path.join(log_path, 'audio_prediction.log'),
                        filemode='w',
                        level=logging.INFO)

    logging.info('Predicting...')

    # Load audio data
    audio_data = librosa.load(file_path, sr=None)

    # FEATURE ENGINEERING
    engineer = FeatureEngineer()

    # Process the audio data
    signal = engineer.feature_engineer(audio_data)

    # MAKE PREDICTION
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        with open(model_path, 'rb') as fp:
            model = pickle.load(fp)

    predictor = BabyCryPredictor(model)
    prediction = predictor.classify(signal)

    logging.info('Prediction: {}'.format(prediction))

    # Check if the prediction indicates a baby cry
    if prediction == "baby_cry":
        print("Baby crying detected! Playing soothing song...")
        # Play soothing song
        if os.name == 'nt':  # Windows
            os.system(f'start /min wmplayer "{johnny.mp3}"')
        # elif os.name == 'posix':  # macOS/Linux
        #     os.system(f'open "{song_path}"')  # macOS
        #     # or
        #     # os.system(f'xdg-open "{song_path}"')  # Linux
    else:
        print("No baby crying detected.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=5, help='Duration of each audio clip in seconds')
    parser.add_argument('--record_interval', type=int, default=10, help='Interval between recordings in seconds')
    parser.add_argument('--model_path', required=True, help='Path to the trained model')
    parser.add_argument('--output_path', default='./recordings', help='Path to save the recorded audio clips')
    parser.add_argument('--log_path', default='./logs', help='Path to save the log files')

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    while True:
        # Generate a unique file name based on timestamp
        timestamp = int(time.time())
        file_path = os.path.join(args.output_path, f"recording_{timestamp}.wav")

        # Record audio
        record_audio(file_path, args.duration, args.output_path)

        # Predict and play if baby crying detected
        predict_and_play(file_path, args.model_path, args.output_path, args.log_path)

        # Wait for the next recording interval
        time.sleep(args.record_interval)

if __name__ == '__main__':
    main()

