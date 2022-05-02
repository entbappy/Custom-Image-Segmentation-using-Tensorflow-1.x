# Image Segmentation using Tensorflow-1.x

This section of the topic helps you to do the Image Segmentation using Tensorflow-1.x . 

1 ) This particular Github repository is made and customized from the original tensorflow website : https://github.com/tensorflow/models/tree/v1.13.0

2 ) Remember you don't have to select current tag version-2.x.0 from original tensorflow website , rather to use version-v1.13.0 

3 ) If you clone the official Tensorflow repository and use it you may face various issues , so to ease out the pain  you can clone this customized repository and use it :  https://github.com/entbappy/Custom-Image-Segmentation-using-Tensorflow-1.x


## Pre-requisites

1 . Python 3.6 ( Recommended )

2 . Tensorflow-1.13.0


## Authors

- [@Jateendra ] (https://github.com/Jateendra)
- [@Bappy ] (https://github.com/entbappy)
## Folder and Files

```bash
1 . data : 
	- This folder contains Imanges and their corressponding json files .
	- The image segementation annotation can be carried out using "labelme" tool . ( Refer : https://github.com/wkentaro/labelme )

2 . env :
	- You can ignore this and can create your own environment in your local system .
	
3 . models :
	- This is a local package which has many folders and files .
	
4 . requirements.txt :
	- Libraries and packages list
	- For local package setup below command is mentioned . This will identify the model and install as local package .
		### local packages -
		-e .

5 . setup.py :
	- this file is required for the models folder .
	- Helps to import models folder .

6 . command.txt :
	- This file is for reference purpose and 2 commands ( one on training and second on referencing are ) mentioned .
	
7 . README.md :
	- Instruction to use this repository .

```


## Setup Procedure

1 . Download/Clone the repository ( https://github.com/entbappy/Custom-Image-Segmentation-using-Tensorflow-1.x )  and open it in VSCode .

2 . Create your conda environment : 
	
	conda create -n myTF1x python=3.6
	conda activate myTF1x
	

	If you have already created "env" environment then activate it as :
	
	conda activate ./env/

3 . Keep only file "labelmapdog.pbtxt" and remove all other files from location : models\research\object_detection\data



## Data Preparation Procedure

1 . Open file "create_tf_records.py" under locatoin : models\research\

2 . Uncomment #training data codes ( Row # 237-240 ) and comment #test data codes ( Row # 243 - 246 ) . Save the file .



3 . Fire below commands in VSCode :

	cd models/research
	pwd
	python create_tf_records.py
	
4 . Comment #training data codes ( Row # 237-240 ) and Uncomment #test data codes ( Row # 243 - 246 ) . Save the file .	

5 . Fire below commands in VSCode :

	python create_tf_records.py

6 . Below 2 files will be getting created under folder : models\research\object_detection\data

	test_dog.record
	train_dog.record


## Execution Procedure

1 . You can download file "mask_rcnn_inception_v2_coco" from location : https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/detection_model_zoo.md and place it under location : "models\research\mask" . But here the required files are present , so you don't need to take any action .

2 . Under folder location : "models\research\mask_training" , below 2 files to be present :

	labelmapdog.pbtxt
	mask_rcnn_inception_v2_coco.config
	
3 . One copy of file "labelmapdog.pbtxt" on location :"models\research\mask_training" to be present under location "models\research\data" .
4 . Open file "models\research\mask_training\mask_rcnn_inception_v2_coco.config" and update as below :

	num_classes: 1 ( Row # 10 )
	fine_tune_checkpoint: "mask/model.ckpt" ( Row # 127 )
	num_steps: 10 ( Row # 133 )
	input_path: "object_detection/data/train_dog.record" ( Row # 142 )
	label_map_path: "object_detection/data/labelmapdog.pbtxt" ( Row # 144 )
	input_path: "object_detection/data/test_dog.record" ( Row # 158 ) 
	label_map_path: "object_detection/data/labelmapdog.pbtxt" ( Row # 160 )
	
5 . Run below commands in VSCode :

	#train cmd: ( Files will be getting generated under location : "models\research\mask_training" )
	python train.py --logtostderr --train_dir=mask_training/ --pipeline_config_path=mask_training/mask_rcnn_inception_v2_coco.config


	#inference cmd ( Files will be getting generated under location : "models\research\inference_graph" )
	python export_inference_graph.py --input_type image_tensor --pipeline_config_path mask_training/mask_rcnn_inception_v2_coco.config --trained_checkpoint_prefix mask_training/model.ckpt-10 --output_directory inference_graph
		
6 . You can find your final model under location : models\research\inference_graph\saved_model\saved_model.pb		
	
		You can use this model for webapplication development .
