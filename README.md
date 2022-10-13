# Plant-disease-explanation
Explanation of citrus disease detection with deep learning

1- Due to massive size of the files (the weights of the customized ResNet50 model calculated for reusability, train set, and test set (numpy arrays) are available at  https://drive.google.com/drive/folders/1_CC8PAPPy9TEaaSVTTfUcEPgWk3O8bXx?usp=sharing

Generally you can follow the instructions below for implementation of steps 1&2:

Step 1) There are either two ways to run  step 1 for PlantDisease.py file.

First, the Original dataset is available at https://data.mendeley.com/datasets/3f83gxmv57/2  based on https://www.sciencedirect.com/science/article/pii/S2352340919306948
You can divide the data through stratified data split into Train,Validation, and Test. In that case you can locally assign the address of Train, Validation, and Test folders in lines 44-46. Then, you can run the lines 51-203 for oversampling, augmentation, and generating train_generator, valid_generator, and test_generator.
The ResNet50 customized model can be both defined and compiled in lines 209-230. The class weights can be computed and the ResNet50 model can be trained in lines 234-264. Finally, the best weight for reusability can be saved and the yhat and the accuracy of train_generator, valid_generator, and test_generator can be calculated.

Second, you can use and upload X1_train, y1_train, X1_test, y1_test (they are npy files extracted based on stratified data split). In that case, you don't need to run lines 44-203 and 228-264. You only need to define and compile the training model in lines 209-222 and upload the previously saved weights (hdf5 file). Don't forget to update the address in line 272. Finally, run line 272 and 274 to calculate the yhat. 


Step 2) You can run lines 320-362 for LIME, Grad-CAM, and SHAP explanation

Additionally, Lines 396-403 calculate the confusion matrix, precision, recall, and f1_score. Lines 281-296 illustrate the model loss and model accuracy only if you followed the first option in step 1. Lines 302-314 generate the X1_train, y1_train, X1_test, y1_test only if you followed the first option in step 1. Lines 368-393 depict the execution time of explanation methods.
