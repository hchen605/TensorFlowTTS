import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

# initialize fastspeech2 model.
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")


# initialize mb_melgan model
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")


# inference
processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")
i = 0
with open('timit_transcript.txt') as f:
    for line in f:
        # For Python3, use print(line)
        if i < 100:
            line = line[2:]
        elif 100 <= i < 1000:
            line = line[3:]
        elif 1000 <= i:
            line = line[4:]
        print(line)

        input_ids = processor.text_to_sequence(line)
        # fastspeech inference

        mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([3,4,5], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
            energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
        )

        # melgan inference
        audio_before = mb_melgan.inference(mel_before)[0, :, 0]
        audio_after = mb_melgan.inference(mel_after)[0, :, 0]

        # save to file
        sf.write('./crisa_timit/'+str(i)+'.wav', audio_after, 22050, "PCM_16")
        #sf.write('./crisa_timit/audio_after.wav', audio_after, 22050, "PCM_16")
        i += 1