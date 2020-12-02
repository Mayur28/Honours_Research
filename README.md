# <b><u>Low-light Image Enhancement using an Unsupervised Generative Adversarial Network</u></b>

We present our solution to the low-light image enhancement problem which is a Generative Adversarial Network (GAN) that is trained in an unsupervised setting. Due to the absence of paired training data, we rely on a series of self-regularization techniques to guide the training process. As a result of training the model in an unsupervised setting, our solution is applicable to images from a broad domain and achieves superior results. We present a handful of representative examples below for your perusal.<br>
<p align="center">
  <img src="https://github.com/Mayur28/MyResearch/blob/master/Representative%20Examples/RepresentativeExamples.png" width="700" height = "400">
</p>

### <b>Environment Setup </b>

We developed this solution in the Google Colab environment as a result of the extensive hardware resources required. In an attempt to enhance the user-friendliness of our implementation, we configured our Colab notebook to use the Nvidia Tesla P100 GPU (with 16GB graphics RAM) which is the most powerful GPU available in the Google Colab environment. Furthermore, the user is not required to explicitly install any packages as a vast majority are pre-installed in the runtime.<br><br>

To execute our implementation, please follow the instructions listed below:<br>
1. Visit https://colab.research.google.com<br>
2. Navigate to File <img src="https://latex.codecogs.com/gif.latex?\rightarrow"/>  Upload Notebook and upload the notebook titled <i>Solution.ipynb</i><br>
3. Since data/ results produced during a session are not persistent (i.e. all data produced during a session is deleted once the runtime is disconnected), we rely on Google Drive for all storage purposes. <br>
4. Should you wish to only use the pre-trained model:<br>
a.    Visit https://drive.google.com/<br>
b.    Breate a directory named <i>Low-light_Image_Enh</i><br>
c.    Within this newly created directory, create an additional directory titled <i>TheModel</i><br>
d.    Upload the pretrained ![pretrained-weights](https://drive.google.com/file/d/1vXTV7TNSSNkrkDtfwyrJ99hPz5i1511D/view?usp=sharing) inside this new directory<br>
5. Return to the Colab notebook and follow the additional instructions provided which explains how the training and testing procedures can be configured to suit your requirements.<br>




