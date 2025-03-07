import tensorflow_hub as hub
import tensorflow as tf

model_url = "https://tfhub.dev/google/translate_en_vi/1"

translator = hub.load(model_url)

def translate_text(text):
    input_text = tf.constant([text])
    translated = translator(input_text)
    return translated.numpy()[0].decode("utf-8")

print(translate_text("Hello, how are you?"))
