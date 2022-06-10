# Code_Challange_for_FellowShip_AI

## Goal:

To use pre-trained ResNet34 on the Stanford Cars dataset, by implementing MixMatch.

## Sources Used for this Project:

#### Pretrained Model:

  1) https://pytorch.org/vision/stable/models.html#torchvision.models.resnet34

  2) https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

#### Kaggle Dataset:
  3) https://ai.stanford.edu/~jkrause/cars/car_dataset.html
 
#### MixMatch PyTorch:

  4) https://github.com/YU1ut/MixMatch-pytorch

  5) https://github.com/facebookresearch/mixup-cifar10

## Results (After training for 40 epochs):

### Accuracy 
  
  By validating on entire validation dataset: 83%
  
### Trained Model (PTH File)

  https://drive.google.com/file/d/11zNGW2F-me8cR0KXt02pQo64zo8DGA0L/view?usp=sharing
  
### Training Loss and Accuracy Graphs

  ![training_loss](https://user-images.githubusercontent.com/33520288/173075615-823b9756-5022-4a4e-8e5b-d545220249df.png)
  
  ![training_accuracy](https://user-images.githubusercontent.com/33520288/173075657-0f18f2f1-c098-497d-bd97-7e49218f0353.png)

### Validation Loss and Accuracy Graphs

  ![validation loss](https://user-images.githubusercontent.com/33520288/173075806-dbd3cb8f-4e94-43d6-b72d-96bdbd79c449.png)

  ![validation acc](https://user-images.githubusercontent.com/33520288/173075833-a73a194e-d51d-401c-9632-855d36229f32.png)

## Usage of the Repo:

### Preparing the data:
  
  Need to provide the paths for the downloaded stanford cars dataset and the annotation labels files for both training and validation dataset.
  
### Training the model:

  ```bash
  $ python3 train.py --batch-size {BATCH_SIZE} --epochs {EPOCH_NUMBER}
  ```
  * For each epoch, a PTH file with the congif file would be generated and saved in the logs folder (Will be created, if there wasn't). The same files will be used for validating the model.
 
### Validating the model:

  ```bash
  $ python3 test.py --config {PATH_TO_THE_GENERATED_CONFIG_FILE} --imgsize {IMAGE_SIZE_FOR_VALIDATION}
  ```
  * In the config file, provide the path for the model, which you want to be loaded.
