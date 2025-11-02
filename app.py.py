"""
===============================================================
Snoonu Thermal Bag Quality Checker
===============================================================
Author: Reem Hashir
Date: November 2025

Project Description:
This project builds an AI-powered image classification tool that determines
whether a courier's thermal delivery bag is in *good* or *bad* condition based
on a photo. Due to limited initial data, the script includes automatic data
augmentation to enhance model training. 

How It Works:
1. The dataset is organized into two folders:
       dataset/
         ├── good/  → images of good bags
         └── bad/   → images of damaged or poor-condition bags

2. The script automatically performs data augmentation (rotation, zoom,
   brightness, flips, etc.) to increase dataset size and improve accuracy.

3. A pre-trained MobileNetV2 model (transfer learning) is fine-tuned to
   classify images as "Good" ✅ or "Bad" ❌.

4. The model is trained with callbacks (early stopping, learning rate reduction)
   and saved for reuse.

5. The final model is deployed using a Gradio web interface styled in
   Snoonu’s red branding, allowing users to upload an image and instantly
   receive a quality prediction.

Technologies Used:
- TensorFlow / Keras (MobileNetV2 + Transfer Learning)
- NumPy & PIL for image handling
- Gradio for web deployment
- ImageDataGenerator for data augmentation

Outcome:
The app provides a fast, user-friendly interface for assessing courier
thermal bag quality, supporting operational efficiency and brand consistency.
===============================================================
"""




import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import gradio as gr

# ============================================================
# 1. Setup dataset folders
# ============================================================
base_dir = "dataset"
train_good = os.path.join(base_dir, "good")
train_bad = os.path.join(base_dir, "bad")
os.makedirs(train_good, exist_ok=True)
os.makedirs(train_bad, exist_ok=True)

MODEL_PATH = "thermal_bag_model.h5"

# ============================================================
# 2. Auto-augment existing images if too few
# ============================================================
def augment_images_if_needed(good_dir, bad_dir, min_count=50, augment_count=100):
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=(0.7, 1.3),
        fill_mode='nearest'
    )

    def augment_class_images(folder):
        images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if len(images) == 0:
            print(f"⚠️ No images found in {folder}. Add at least one image.")
            return
        if len(images) >= min_count:
            print(f"✅ Enough images in {folder}, skipping augmentation.")
            return
        img_path = os.path.join(folder, images[0])
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        prefix = os.path.splitext(os.path.basename(img_path))[0]
        print(f"🧩 Augmenting images for {folder}...")
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=folder,
                                  save_prefix=prefix, save_format='jpg'):
            i += 1
            if i >= augment_count:
                break

    augment_class_images(good_dir)
    augment_class_images(bad_dir)

augment_images_if_needed(train_good, train_bad)

# ============================================================
# 3. Prepare data generators
# ============================================================
batch_size = 8
img_size = (224, 224)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# ============================================================
# 4. Build model
# ============================================================
if os.path.exists(MODEL_PATH):
    print(" Loading existing trained model...")
    model = load_model(MODEL_PATH)
else:
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model initially
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, monitor='val_loss')
    ]

    # ============================================================
    # 5. Train and fine-tune
    # ============================================================
    if train_generator.samples > 10:
        steps_per_epoch = max(1, train_generator.samples // batch_size)
        validation_steps = max(1, val_generator.samples // batch_size)

        print("Training base model...")
        model.fit(
            train_generator,
            epochs=10,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks
        )

        print("Fine-tuning top layers...")
        for layer in model.layers[-40:]:
            layer.trainable = True

        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(
            train_generator,
            epochs=10,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks
        )

        model.save(MODEL_PATH)
    else:
        print("Not enough training data. Add more images to 'dataset/good' and 'dataset/bad'.")

# ============================================================
# 6. Prediction function
# ============================================================
try:
    class_map = {v: k for k, v in train_generator.class_indices.items()}
except Exception:
    class_map = {0: "bad", 1: "good"}

def classify_bag(img):
    if img is None:
        return "No image"
    img = img.resize(img_size)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0]
    idx = int(np.argmax(pred))
    label = class_map.get(idx, str(idx))
    confidence = float(np.max(pred))
    return {label: confidence}

# ============================================================
# 7. Gradio Interface (Snoonu branding)
# ============================================================
custom_css = """
body {
    background-color: #d6001c;
    font-family: 'Poppins', sans-serif;
    color: white;
}
.gradio-container {
    background-color: #d6001c !important;
}
h1, h2, p {
    color: white !important;
    text-align: center;
}
"""

logo_path = "snoonu_logo.png"  # Place your Snoonu logo file in the same directory

iface = gr.Interface(
    fn=classify_bag,
    inputs=gr.Image(type="pil", label="📸 Upload your thermal bag image"),
    outputs=gr.Label(num_top_classes=1, label="Prediction"),
    title="📦 Snoonu Thermal Bag Quality Checker",
    description="Upload a photo of a Snoonu courier thermal bag. The model predicts if it's **Good ✅** or **Bad ❌**.",
    article=f'<img src="{logo_path}" width="200">',
    css=custom_css
)

if __name__ == "__main__":
    iface.launch(share=True)
