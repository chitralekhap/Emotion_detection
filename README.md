# Emotion Detection from Images

This project is designed to detect emotions from images in a given folder using a convolutional neural network (CNN) trained on facial expression data. The detected emotions are displayed on the image, including the emotion name in the top-right corner of the image.

## Project Structure

- `recognize_image.py`: Main script for training and running the emotion detection.
- `data/`: Folder containing training (`data/train`) and validation (`data/test`) data.
- `haarcascade_frontalface_default.xml`: Haar Cascade file for face detection.
- `model.h5`: Saved model weights after training.
- `plot.png`: Graph of accuracy and loss (generated after training).

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

You can install the necessary libraries using:

pip install tensorflow keras opencv-python numpy matplotlib
Usage
1. Training the Model
To train the model, use the following command:

python recognize_image.py --mode train
This will train the CNN using the data in data/train and data/test and save the model weights to model.h5.

2. Display Emotion Detection
To detect emotions from images in a folder:

Place your images in a specified folder (e.g., path/to/your/image/folder).

python recognize_image.py --mode display
The script will read each image, detect faces, predict the emotion, and display it with the emotion name shown in the top-right corner of the image. Press any key to move to the next image.

Code Overview
Model Creation: A CNN with multiple Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers.
Training: The model is compiled with Adam optimizer and categorical_crossentropy loss function.
Emotion Detection:
Uses a Haar Cascade to detect faces.
Processes each detected face and makes emotion predictions using the loaded model weights.
Displays the predicted emotion name at the top-right corner of each image.
Emotion Labels
The model can detect the following emotions:

Angry
Disgusted
Fearful
Happy
Neutral
Sad
Surprised
Customization
Adjust Image Folder: Modify the image_folder variable in the code to point to the folder containing your images.
Positioning Text: The coordinates for the top-right display of emotions can be adjusted for different image sizes.
Example Output
When running in display mode, the script will show images with bounding boxes around detected faces and the predicted emotion displayed both above the face and in the top-right corner.

Future Enhancements
Adding real-time video feed emotion detection.
Integrating with more complex face detection models for better accuracy.
Fine-tuning the model with more diverse datasets for improved emotion recognition.
License
This project is open-source and available under the MIT License.

Acknowledgments
TensorFlow and Keras for providing a powerful framework for deep learning.
OpenCV for face detection and image processing capabilities.
Original datasets for emotion recognition.
javascript
Copy code
