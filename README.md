# Plant-disease-explanation
Explanation of citrus disease detection using Multi-Objective Genetic Algorithm Explainer (MOGAE)

The corresponding paper is UNDER REVIEW.

There are two .py files where the pipeline for developing the fine-tuned ResNet5 is implemented in __PlantDisease.py__ and the pipeline for developing MOGAE is implemented in __MOGAE.py__. Hence, the instructions for each file are provided separately as follows:



__NOTICE__ ==> Due to massive size of the files (the weights of the customized ResNet50 model calculated for reusability, X1_train, y1_train, X1_test, y1_test (numpy arrays) are available [here](https://drive.google.com/drive/folders/1_CC8PAPPy9TEaaSVTTfUcEPgWk3O8bXx?usp=sharing)


__Step 1)__  __PlantDisease.py__.

There are either two ways to run  step 1 for __PlantDisease.py__ file. First, the Original dataset is available [here](https://data.mendeley.com/datasets/3f83gxmv57/2)  based on the [corresponding paper](https://www.sciencedirect.com/science/article/pii/S2352340919306948). You can divide the data through stratified data split into Train,Validation, and Test. In that case you can locally assign the address of Train, Validation, and Test folders in lines 44-46. Then, you can run the lines 51-203 for oversampling, augmentation, and generating train_generator, valid_generator, and test_generator. The ResNet50 customized model can be both defined and compiled in lines 209-230. The class weights can be computed and the ResNet50 model can be trained in lines 234-264. Finally, the best weight for reusability can be saved and the yhat and the accuracy of train_generator, valid_generator, and test_generator can be calculated.

Second, you can utilize and upload X1_train, y1_train, X1_test, and y1_test. These files are in .npy format and were extracted using a stratified data split, which precisely aligns with the results presented in the paper. In that case, you don't need to run lines 44-203 and 228-264. You only need to define and compile the training model in lines 209-222 and upload the previously saved weights (hdf5 file). Don't forget to update the address in line 272. Finally, run line 272 and 274 to calculate the yhat. 

__Step 2)__  __MOGAE.py__ .

Assuming the model has already been defined and compiled, the instructions for MOGAE are provided as follows:

Run Lines 1-293 for the corresponding functions initially. Next, execute the systematic search from lines 295 to 357. Ensure to define the test image to be explained in line 299, which by default is set to 5, indicating that the 5th image in the test sample will be the subject of explanation. Next, execute lines 359-623 for the main body of MOGAE. Subsequently, run lines 631-660 to extract _optimal images_ from the Pareto front. Proceed with executing lines 672-685 for majority voting and generating the _MV image_. Following this, the _explanation image_, an intersection of the _base image_ and _MV image_, can be generated from lines 707-724. Lastly, lines 733 until the end encompass codes for generating plots, normalized explanation error, LIME, and more.

For the purpose of reusability:

__ATTENTION 1:__  The images that have been used in the paper are 5th(Black spot), 29th(Canker), 46th(Greening), and 55th(Healthy) image from X1_test based on our split.

__ATTENTION 2:__ The delineated images by SLIC algorithm are image5_delineated.npy(Black spot), image29_delineated.npy(Canker), image46_delineated.npy(Greening), and image55_delineated.npy(Healthy). You can use them in lines 759-771 for calculating normalized error of explanation by initializing the intended _delineation_ variable in line 761. _black_back_ is the _explanation image_.

__ATTENTION 3:__ You can also calculate the maximum error of explanation for each image using worst5.npy(Black spot), worst29.npy(Canker), worst46.npy(Greening), and worst55.npy(Healthy). You can use them in lines 759-771 for respective calculations. by initializing the intended _worst_ variable in line 760.

<!-- Step 2) You can run lines 320-362 for LIME, Grad-CAM, and SHAP explanation

Additionally, Lines 396-403 calculate the confusion matrix, precision, recall, and f1_score. Lines 281-296 illustrate the model loss and model accuracy only if you followed the first option in step 1. Lines 302-314 generate the X1_train, y1_train, X1_test, y1_test only if you followed the first option in step 1. Lines 368-393 depict the execution time of explanation methods.

The four selected test samples in the paper (image id 1: Black spot, image id 2: Canker, image id 3: Greening, image id 4: Healthy) are the 4th, 21st, 51st, and 55th samples in the test set (X1_test)-->
