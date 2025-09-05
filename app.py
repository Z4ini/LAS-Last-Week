import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import time
from transformers import BertTokenizer

# st.set_page_config(page_title="Multi-Page AI App", layout="wide")

def home():
    st.title("MBC Lab Recruitment: Final Project Showcase")
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrpKeg8H46R3QFGB8Y-ZCxX3arlbv-UUNu5g&s")
    st.write("Welcome to my final project for the Learning Advanced Skill (LAS) program, the 5-week recruitment phase for the Multimedia, Big Data, Cybersecurity (MBC) Lab at Telkom University.")
    st.write("This web application serves as a demonstration of the deep learning skills acquired throughout the LAS program. It features three distinct models, each tackling a unique challenge in the fields of computer vision and natural language processing. The models were developed during Week 3 and Week 4 and are now deployed here for interactive use.")
    st.write("Please explore each project to see the models in action. Thank you for the incredible learning opportunity provided by the MBC Lab.")
    
def cats_vs_dogs_classifier():
    st.title("Cats Vs Dogs Classifier")
    st.image("https://www.watchmojo.com/uploads/thumbs720/LL-Cats-VS-Dogs-720p30.jpg", caption="Cats or Dogs")
    st.header("Project Overview:")
    st.write("This model is designed to solve a classic computer vision problem: distinguishing between images of cats and dogs. It analyzes an uploaded photo and predicts which of these two animals is present. This project, developed in Week 3, serves as a fundamental exercise in building and training an image classification model.")
    
    
    st.header("Technical Details:")
    st.write("The model is a Convolutional Neural Network (CNN), which is the standard architecture for image recognition tasks. (Note: While you mentioned RNN, CNNs are the correct architecture for image classification. Using this term will be more accurate for your project description). The network has been trained to recognize key features, patterns, and textures—such as whiskers, ear shape, and snout—to make an accurate prediction.")

    model_path = hf_hub_download(repo_id="Zainiiii/CatsnDogs-Model", filename="best_model_catsndogs.keras")
    model = tf.keras.models.load_model(model_path)

    st.subheader("Try It Out!")

    uploaded_file = st.file_uploader("Upload Gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:

        if st.button("RUN THE MODEL"):
            with st.spinner("Please wait, the model is predicting..."):
                time.sleep(2)

                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded Image.', use_container_width=True)

                img_resized = image.resize((64, 64))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = img_array / 255.0
                img_batch = np.expand_dims(img_array, axis=0)
        

                prediction = model.predict(img_batch)

                score = prediction[0][0]
                if score < 0.5:
                    st.subheader("It's A Dog")
                else:
                    st.subheader("It's A Cat")




def food_101_classifier():

    from keras.models import Sequential
    from keras.layers import Rescaling, GlobalAveragePooling2D, Dense, Dropout
    from keras.regularizers import l2

    def create_model_architecture():
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None  # We use None because we will load our own trained weights
        )
        # Ensure the base_model is not trainable if it was frozen during training
        base_model.trainable = False

        model = Sequential([
            # Note: The Rescaling layer is part of the model, so we don't need to divide by 255.0 later
            Rescaling(1./255, input_shape=(224, 224, 3)),
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(101, activation='softmax')
        ])
        return model

    # --- Part 2: Load the Model and Weights Efficiently ---
    # This function downloads the weights file and is cached so it only runs once.
    @st.cache_data
    def get_weights_path():
        model_path = hf_hub_download(repo_id="Zainiiii/food_101", filename="best_model_101_v2.keras")
        return model_path

    # This function creates the architecture and loads the weights into it.
    # It's cached as a resource to prevent rebuilding/reloading on every interaction.
    @st.cache_resource
    def load_complete_model():
        model = create_model_architecture()
        model.load_weights(get_weights_path())
        return model
        

    class_labels = [
            'Apple pie', 'Baby back ribs', 'Baklava', 'Beef carpaccio', 'Beef tartare',
            'Beet salad', 'Beignets', 'Bibimbap', 'Bread pudding', 'Breakfast burrito',
            'Bruschetta', 'Caesar salad', 'Cannoli', 'Caprese salad', 'Carrot cake',
            'Ceviche', 'Cheesecake', 'Cheese plate', 'Chicken curry', 'Chicken quesadilla',
            'Chicken wings', 'Chocolate cake', 'Chocolate mousse', 'Churros', 'Clam chowder',
            'Club sandwich', 'Crab cakes', 'Creme brulee', 'Croque madame', 'Cup cakes',
            'Deviled eggs', 'Donuts', 'Dumplings', 'Edamame', 'Eggs benedict', 'Escargots',
            'Falafel', 'Filet mignon', 'Fish and chips', 'Foie gras', 'French fries',
            'French onion soup', 'French toast', 'Fried calamari', 'Fried rice',
            'Frozen yogurt', 'Garlic bread', 'Gnocchi', 'Greek salad', 'Grilled cheese sandwich',
            'Grilled salmon', 'Guacamole', 'Gyoza', 'Hamburger', 'Hot and sour soup',
            'Hot dog', 'Huevos rancheros', 'Hummus', 'Ice cream', 'Lasagna', 'Lobster bisque',
            'Lobster roll sandwich', 'Macaroni and cheese', 'Macarons', 'Miso soup',
            'Mussels', 'Nachos', 'Omelette', 'Onion rings', 'Oysters', 'Pad thai', 'Paella',
            'Pancakes', 'Panna cotta', 'Peking duck', 'Pho', 'Pizza', 'Pork chop', 'Poutine',
            'Prime rib', 'Pulled pork sandwich', 'Ramen', 'Ravioli', 'Red velvet cake',
            'Risotto', 'Samosa', 'Sashimi', 'Scallops', 'Seaweed salad', 'Shrimp and grits',
            'Spaghetti bolognese', 'Spaghetti carbonara', 'Spring rolls', 'Steak',
            'Strawberry shortcake', 'Sushi', 'Tacos', 'Takoyaki', 'Tiramisu', 'Tuna tartare',
            'Waffles']

    st.title("Food-101 Classifier")
    st.image("https://rp-cms.imgix.net/wp-content/uploads/AdobeStock_513646998-scaled.jpeg", caption="Food-101 Dataset")
    st.header("Project Overview:")
    st.write("Taking image classification to the next level, this model tackles a significantly more complex challenge: identifying 101 different types of food from a photograph. From sushi to cheesecake, this model demonstrates the ability to differentiate between a large number of closely related categories. This was the second project from Week 3, focusing on large-scale classification.")
    
    
    st.header("Technical Details:")
    st.write("This model is also built upon a Convolutional Neural Network (CNN) architecture. Given the complexity of distinguishing between 101 classes, this project likely utilizes a technique called transfer learning. This involves using a powerful, pre-trained model (like EfficientNet or ResNet) as a foundation and fine-tuning it on the Food-101 dataset. This approach leverages existing knowledge to achieve high accuracy on a difficult task.")

    # model_path = hf_hub_download(repo_id="Zainiiii/food_101", filename="best_model_101_v3.keras")
    # model.load_weights(load_food_101_model())
    # model = tf.keras.models.load_model(model_path)
    model = load_complete_model()

    st.subheader("Try It Out!")

    uploaded_file = st.file_uploader("Upload Gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        if st.button("RUN"):
            with st.spinner("Please wait, the model is predicting..."):
                time.sleep(2)

                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded Image.', use_container_width=True)

                img = image.resize((224, 224))
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                prediction = model.predict(img)
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                predicted_class_name = class_labels[predicted_class_index]

                confidence_score = np.max(prediction) * 100

                st.success(f"Prediction: **{predicted_class_name}**")
                st.info(f"Confidence: **{confidence_score:.2f}%**")

def sentiment_analysis():
    st.title("Sentiment Analysis")
    st.image("https://media.licdn.com/dms/image/v2/D4E12AQH27JACHVekzw/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1693135940957?e=2147483647&v=beta&t=Tm2IuaolXy-SzE28UoBKawuB-7RV24r5lLApasGOj1g", caption="Social media sentiment")
    st.header("Project Overview:")
    st.write("This model, developed in Week 4, moves from images to text. It is designed to analyze the sentiment of a given sentence, such as a comment from Twitter or Instagram. The model determines whether the emotional tone of the text is Sadness, Anger, Support, Hope, or Dissapointment. This is a core task in Natural Language Processing (NLP) with wide applications.")

    st.header("Technical Details:")
    st.write("This model is built using a Transformer-based architecture. Unlike older models, Transformers are exceptionally good at understanding the context and nuanced relationships between words in a sentence, making them state-of-the-art for many NLP tasks. The model has been fine-tuned on a dataset of social media comments to learn the patterns associated with different sentiments.")

    model_path = hf_hub_download(repo_id="Zainiiii/sentiment_model", filename="model_TomLembong_quantized.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    st.subheader("Try It Out!")
    st.write('Enter a sentence, comment, or tweet into the text box below to see its predicted sentiment.')

    input_text = st.text_input("")

    if st.button("Analyze Sentiment"):
        if input_text:
            with st.spinner("Analyzing the sentiment..."):
                time.sleep(2)

                tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
                tokenized_input = tokenizer(
                text=input_text,
                max_length=250,      
                padding='max_length',
                truncation=True,
                return_tensors='np'
                )

                input_ids = tokenized_input['input_ids'].astype(np.int32)
                attention_mask = tokenized_input['attention_mask'].astype(np.int32)

                interpreter.set_tensor(input_details[0]['index'], attention_mask)
                interpreter.set_tensor(input_details[1]['index'], input_ids)

                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])

                probabilities = tf.nn.softmax(output_data)[0].numpy()

                predicted_index = np.argmax(probabilities)

                class_labels = ['SADNESS', 'ANGER', 'SUPPORT', 'HOPE', 'DISAPPOINTMENT']

                final_prediction = class_labels[predicted_index]

                if final_prediction == 'SADNESS':
                    st.badge("SADNESS", color="violet")
                    st.write(f"Kalimat:  {input_text}")
                    # st.write(f"Prediksi:  {final_prediction}")
                    # st.write(f"Keyakinan:  {probabilities[predicted_index]:.2%}")
                    
                elif final_prediction == 'ANGER':
                    st.badge("ANGER", color="red")
                    st.write(f"Kalimat:  {input_text}")
                    # st.write(f"Prediksi:  {final_prediction}")
                    # st.write(f"Keyakinan:  {probabilities[predicted_index]:.2%}")

                elif final_prediction == 'SUPPORT':
                    st.badge("SUPPORT", color="blue")
                    st.write(f"Kalimat:  {input_text}")
                    # st.write(f"Prediksi:  {final_prediction}")
                    # st.write(f"Keyakinan:  {probabilities[predicted_index]:.2%}")

                elif final_prediction == 'HOPE':
                    st.badge("HOPE", color="green")
                    st.write(f"Kalimat:  {input_text}")
                    # st.write(f"Prediksi:  {final_prediction}")
                    # st.write(f"Keyakinan:  {probabilities[predicted_index]:.2%}")

                elif final_prediction == 'DISAPPOINTMENT':
                    st.badge("DISAPPOINTMENT", color="grey")
                    st.write(f"Kalimat:  {input_text}")
                    # st.write(f"Prediksi:  {final_prediction}")
                    # st.write(f"Keyakinan:  {probabilities[predicted_index]:.2%}")






pages = {
    "Home": [
        st.Page(home, title="Welcome", icon="🏠", default=True)
    ],
    "Models": [
        st.Page(cats_vs_dogs_classifier, title="Cats vs Dogs", icon="🐾"),
        st.Page(food_101_classifier, title="Food-101", icon="🍔"),
        st.Page(sentiment_analysis, title="Sentiment Analysis", icon="🤖"),
    ]
}

pg = st.navigation(pages)
pg.run()
