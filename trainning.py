from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from keras.preprocessing.image import ImageDataGenerator
import os

# Set the image dimensions
img_width, img_height = 128, 128

# Create the KNN classifier
classifier = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))

# Set the path to the training and validation data directories
train_data_dir = 'C:/PBL SE/Plant-Leaf-Disease-Prediction-main/Dataset/train'
valid_data_dir = 'C:/PBL SE/Plant-Leaf-Disease-Prediction-main/Dataset/val'

# Set the batch size and number of classes
batch_size = 6
num_classes = len(os.listdir(train_data_dir))

# Set up the image data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess the training and validation data
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(valid_data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical')

# Extract the features and labels from the training data
train_features = train_generator.next()[0]
train_labels = train_generator.classes

# Reshape the feature array for KNN input
n_samples, n_features, _, _ = train_features.shape
train_features = train_features.reshape((n_samples, n_features * img_width * img_height))

# Fit the KNN classifier on the training data
classifier.fit(train_features, train_labels)

# Make predictions on the validation data
valid_features = valid_generator.next()[0]
valid_labels = valid_generator.classes
n_samples_valid = valid_features.shape[0]
valid_features = valid_features.reshape((n_samples_valid, n_features * img_width * img_height))
predictions = classifier.predict(valid_features)

# Print the predictions
print(predictions)
