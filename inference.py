import tensorflow as tf
import SpeechModels
import torchaudio

class Recognizer:
    def __init__(self):
        self.model = SpeechModels.AttRNNSpeechModel(36, samplingrate = 16000, inputLength = None)
        self.classes = ["oragh", "arz", "sekeh", "bank", "tala", "naft", "moshtaghat", "felezat", "sahami", "sabet", "mokhtalet", "ghabel-moamele"]

    def load_model(self, model_path="./model-attRNN-reza.h5"):
        self.model.load_weights(model_path)
    
    def predict(self, audio_path):
        wave, freq = torchaudio.load(audio_path)
        wave = torchaudio.functional.resample(wave, orig_freq=freq, new_freq=16000)
        wave = wave.numpy()[0]
        predict = self.model(wave.reshape([1,-1]))[0,:12]
        predict = list(predict.numpy())
        return {
            name: pred for name, pred in zip(self.classes, predict)
        }
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="audio file path to evaluate")
    args = parser.parse_args()

    model = Recognizer()
    model.load_model()
    print(model.predict(args.path))
