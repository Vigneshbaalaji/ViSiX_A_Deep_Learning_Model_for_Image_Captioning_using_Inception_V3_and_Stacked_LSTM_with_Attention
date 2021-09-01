# ViSiX_A_Deep_Learning_Model_for_Image_Captioning_using_Inception_V3_and_Stacked_LSTM_with_Attention


## OBJECTIVE

<p align='justify'> This project aims to enable efficient translation of images to text using a hybrid model that combines Natural Language Processing and Computer Vision Techniques. This methodology finds its application in a vast range of fields such as, defense, self-driving cars, and social media.</p>

<p align='justify'> The task of image captioning can be divided into two modules logically, namely an image-based model which extracts the features and nuances out of our image, and the other is a language-based model which translates the features and objects given by our image-based model to a natural sentence. In our method, the feature extraction is handled by Inception V3, and the translation is handled by LSTM Recurrent Neural Networks. Our method makes use of Flickr 8k Dataset and Google Firebase Cloud Firestore for storage and retrieval of images. We intend to deploy the model for use through a Streamlit Web application.</p>
  
  
## IMPLEMENTATION METHODOLOGY

<p align='justify'> The Images along with their respective captions are taken from the dataset for training our model. The Images are preprocessed by resizing and normalizing them and the inputs are given to the pre trained Inception V3 model for feature extraction and then these features are fed into the LSTM. Similarly, the captions are also preprocessed to convert sequences of same length. The model architecture is given below.</p>

![alt text](https://github.com/Vigneshbaalaji/ViSiX_A_Deep_Learning_Model_for_Image_Captioning_using_Inception_V3_and_Stacked_LSTM_with_Attention/blob/main/Arch_Diagram.png?raw=true)

### APPROACH-1: IMPLEMENTATION WITH MICROSOFT COMMON OBJECTS IN CONTEXT (MS-COCO) DATASET

<p align='justify'> A sample of 6000 images out of a total of 82,000 images with its corresponding captions from the MS COCO Dataset is used for training our model. After splitting the data into train and validation sets, the images are passed to the pre-trained Inception V3 model for extraction of features. The images are initially preprocessed by resizing and also normalizing.</p>
  
<p align='justify'> Simultaneously, the captions are preprocessed by tokenizing and unique words are obtained. The vocabulary size is reduced to the size of 5000 words followed by creating word-to-index and index-to-word mappings, and all the sequences are padded to the same size. The data is now split into training and testing data. For caption generation, attention-based model is used for caption generation Hence using this attention mechanism, the model is trained. The attention model consists of CNN encoder and RNN decoder. This helps us fetching more contextual information. Then, to capture both the forward and reverse relationship of the sequence, we use Bidirectional LSTM. </p>

### APPROACH-2: IMPLEMENTATION WITH FLICKR 8K DATASET

<p align='justify'> The Flickr 8K dataset consisting of 8000 images are already split into train and test set.  Transfer Learning is incorporated by choosing one of the pre-trained models like Xception, VGG16, ResNet, Inception V3.  Contrast Limited Adaptive Histogram Equalization (CLAHE) is a variant of Adaptive histogram equalization (AHE) which takes care of over-amplification of the contrast. CLAHE operates on small regions in the image, called tiles, rather than the entire image. The neighboring tiles are then combined using bilinear interpolation to remove the artificial boundaries. This algorithm can be applied to improve the contrast of images.</p>
 
<p align='justify'> This further enhances the quality of the image and amplifies the variance in pixels which helps the model to distinguish better between different images. As the images we process are colored images (RGB/BGR images), they do not hold the entire image intensity levels directly. Hence, these images have to be converted into LAB color space format before applying CLAHE, where the image is split into three planes, where L holds the luminosity (light intensity levels) without color, A holds chromatic information where color falls along the red-green axis, and B again holds chromatic information where color falls along the blue-yellow axis. </p>

<p align='justify'> The pretrained Global Vectors for Word Representation (GloVe) embedding are downloaded, and loaded. Among the various versions of GloVe, the GloVe 200-dimension version is considered for this project, where, each word is represented using 200 embedding values, and are converted into GloVe vectors. The extension of Word2Vec algorithm for efficiently learning word vectors is GloVe Algorithm. It is an unsupervised learning algorithm for obtaining vector representations for words. In order to construct an explicit word – context it uses statistics across the whole text corpus. Instead of loading all the images at once, which may cause memory issue, the data generator is used to get around it. It requires all the images are saved in one folder. To apply the generator, we determine the maximal length of the description, and define the batch size as the number of photos to fit into the model each time. The input X has two parts: image features from InceptionV3 of dimensions 2048 x 1 and sequential descriptions with a maximum length of 34. The sequence dimension of 283 based on the number of words in the five descriptions. Remember that we are aiming to have the input as the structure like this. The output Y has a dimension of vocabulary size 8811 because it uses one-hot encoding. And finally, the images and the captions are fed into the LSTM model for image caption generation. </p>
  
## AUDIO GENERATION USING GOOGLE TEXT-TO-SPEECH

<p align='justify'> In addition to the captions generated, the captions are translated from text to speech using gTTS library. gTTS (Google Text-to-Speech), a Python library and CLI tool to interface with Google Translate's Text-to-Speech API. This is specifically integrated into this application so that the audio can be heard to aid the visually impaired. </p>

## OUTPUT SCREENSHOTS

<table style="width:100%">
  
  <tr>
    <th>APPROACH-1: Trained on MS-COCO Dataset</th>
    <th>APPROACH-2: Trained on Flickr8K Dataset</th>
  </tr>
  
  <tr>
    <td><img alt="APPROACH-1: Trained on MS-COCO Dataset" src="https://github.com/Vigneshbaalaji/ViSiX_A_Deep_Learning_Model_for_Image_Captioning_using_Inception_V3_and_Stacked_LSTM_with_Attention/blob/main/MS_COCO_Model_Output.JPG" /> </td>
    <td><img alt="APPROACH-2: Trained on Flickr8K Dataset" src="https://github.com/Vigneshbaalaji/ViSiX_A_Deep_Learning_Model_for_Image_Captioning_using_Inception_V3_and_Stacked_LSTM_with_Attention/blob/main/Flickr_Model_Output.png" /> </td>
  </tr>
