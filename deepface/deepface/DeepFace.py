from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
import time
import os
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID, DlibWrapper, Boosting
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst

def build_model(model_name):
	
	models = {
		'VGG-Face': VGGFace.loadModel, 
		'OpenFace': OpenFace.loadModel,
		'Facenet': Facenet.loadModel,
		'DeepFace': FbDeepFace.loadModel,
		'DeepID': DeepID.loadModel,
		'Dlib': DlibWrapper.loadModel,
		'Emotion': Emotion.loadModel,
		'Age': Age.loadModel,
		'Gender': Gender.loadModel,
		'Race': Race.loadModel
	}

	model = models.get(model_name)
	
	if model:
		model = model()
		#print('Using {} model backend'.format(model_name))
		return model
	else:
		raise ValueError('Invalid model_name passed - {}'.format(model_name))

def verify(img1_path, img2_path = '', model_name = 'VGG-Face', distance_metric = 'cosine', 
		   model = None, enforce_detection = True, detector_backend = 'mtcnn'):	

	tic = time.time()
	
	img_list, bulkProcess = initialize_input(img1_path, img2_path)	
	functions.initialize_detector(detector_backend = detector_backend)

	resp_objects = []
	
	#--------------------------------
	
	if model_name == 'Ensemble':
		return Boosting.verify(model = model, img_list = img_list, bulkProcess = bulkProcess, enforce_detection = enforce_detection, detector_backend = detector_backend)
		
	#ensemble learning block end
	
	#--------------------------------
	#ensemble learning disabled
	
	if model == None:		
		model = build_model(model_name)
	
	#------------------------------
	#face recognition models have different size of inputs
	#my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
		
	input_shape = model.layers[0].input_shape
	
	if type(input_shape) == list:
		input_shape = input_shape[0][1:3]
	else:
		input_shape = input_shape[1:3]
	
	input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

	#------------------------------

	#tuned thresholds for model and metric pair
	threshold = functions.findThreshold(model_name, distance_metric)

	#------------------------------
	
	#calling deepface in a for loop causes lots of progress bars. this prevents it.
	disable_option = False if len(img_list) > 1 else True
	
	pbar = tqdm(range(0,len(img_list)), desc='Verification', disable = disable_option)
	
	#for instance in img_list:
	for index in pbar:
	
		instance = img_list[index]
		
		if type(instance) == list and len(instance) >= 2:
			img1_path = instance[0]
			img2_path = instance[1]

			#----------------------
			#crop and align faces

			img1 = functions.preprocess_face(img=img1_path, target_size=(input_shape_y, input_shape_x), enforce_detection = enforce_detection, detector_backend = detector_backend)
			img2 = functions.preprocess_face(img=img2_path, target_size=(input_shape_y, input_shape_x), enforce_detection = enforce_detection, detector_backend = detector_backend)

			#----------------------
			#find embeddings

			img1_representation = model.predict(img1)[0,:]
			img2_representation = model.predict(img2)[0,:]

			#----------------------
			#find distances between embeddings

			if distance_metric == 'cosine':
				distance = dst.findCosineDistance(img1_representation, img2_representation)
			elif distance_metric == 'euclidean':
				distance = dst.findEuclideanDistance(img1_representation, img2_representation)
			elif distance_metric == 'euclidean_l2':
				distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
			else:
				raise ValueError("Invalid distance_metric passed - ", distance_metric)

			#----------------------
			#decision

			if distance <= threshold:
				identified = True
			else:
				identified = False

			#----------------------
			#response object
			
			resp_obj = {
				"verified": identified
				, "distance": distance
				, "max_threshold_to_verify": threshold
				, "model": model_name
				, "similarity_metric": distance_metric
				
			}
			
			if bulkProcess == True:
				resp_objects.append(resp_obj)
			else:
				return resp_obj
			#----------------------

		else:
			raise ValueError("Invalid arguments passed to verify function: ", instance)

	#-------------------------

	toc = time.time()

	#print("identification lasts ",toc-tic," seconds")
	
	if bulkProcess == True:
		
		resp_obj = {}

		for i in range(0, len(resp_objects)):
			resp_item = resp_objects[i]
			resp_obj["pair_%d" % (i+1)] = resp_item
			
		return resp_obj

def analyze(img_path, actions = [], models = {}, enforce_detection = True
			, detector_backend = 'mtcnn'):

	img_paths, bulkProcess = initialize_input(img_path)
	functions.initialize_detector(detector_backend = detector_backend)
	
	#---------------------------------

	#if a specific target is not passed, then find them all
	if len(actions) == 0:
		actions= ['emotion', 'age', 'gender', 'race']

	#print("Actions to do: ", actions)

	#---------------------------------

	if 'emotion' in actions:
		if 'emotion' in models:
			print("already built emotion model is passed")
			emotion_model = models['emotion']
		else:
			emotion_model = build_model('Emotion')

	if 'age' in actions:
		if 'age' in models:
			#print("already built age model is passed")
			age_model = models['age']
		else:
			age_model = build_model('Age')

	if 'gender' in actions:
		if 'gender' in models:
			print("already built gender model is passed")
			gender_model = models['gender']
		else:
			gender_model = build_model('Gender')

	if 'race' in actions:
		if 'race' in models:
			print("already built race model is passed")
			race_model = models['race']
		else:
			race_model = build_model('Race')
	#---------------------------------

	resp_objects = []
	
	disable_option = False if len(img_paths) > 1 else True
	
	global_pbar = tqdm(range(0,len(img_paths)), desc='Analyzing', disable = disable_option)
	
	#for img_path in img_paths:
	for j in global_pbar:
		img_path = img_paths[j]

		resp_obj = {}
		
		disable_option = False if len(actions) > 1 else True

		pbar = tqdm(range(0,len(actions)), desc='Finding actions', disable = disable_option)

		action_idx = 0
		img_224 = None # Set to prevent re-detection
		#for action in actions:
		for index in pbar:
			action = actions[index]
			pbar.set_description("Action: %s" % (action))

			if action == 'emotion':
				emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
				img = functions.preprocess_face(img = img_path, target_size = (48, 48), grayscale = True, enforce_detection = enforce_detection, detector_backend = detector_backend)

				emotion_predictions = emotion_model.predict(img)[0,:]

				sum_of_predictions = emotion_predictions.sum()

				resp_obj["emotion"] = {}
				
				for i in range(0, len(emotion_labels)):
					emotion_label = emotion_labels[i]
					emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
					resp_obj["emotion"][emotion_label] = emotion_prediction
				
				resp_obj["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]

			elif action == 'age':
				if img_224 is None:
					img_224 = functions.preprocess_face(img = img_path, target_size = (224, 224), grayscale = False, enforce_detection = enforce_detection, detector_backend = detector_backend) #just emotion model expects grayscale images
				#print("age prediction")
				age_predictions = age_model.predict(img_224)[0,:]
				apparent_age = Age.findApparentAge(age_predictions)

				resp_obj["age"] = str(int(apparent_age))

			elif action == 'gender':
				if img_224 is None:
					img_224 = functions.preprocess_face(img = img_path, target_size = (224, 224), grayscale = False, enforce_detection = enforce_detection, detector_backend = detector_backend) #just emotion model expects grayscale images
				#print("gender prediction")

				gender_prediction = gender_model.predict(img_224)[0,:]

				if np.argmax(gender_prediction) == 0:
					gender = "Woman"
				elif np.argmax(gender_prediction) == 1:
					gender = "Man"

				resp_obj["gender"] = gender

			elif action == 'race':
				if img_224 is None:
					img_224 = functions.preprocess_face(img = img_path, target_size = (224, 224), grayscale = False, enforce_detection = enforce_detection, detector_backend = detector_backend) #just emotion model expects grayscale images
				race_predictions = race_model.predict(img_224)[0,:]
				race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']

				sum_of_predictions = race_predictions.sum()
				
				resp_obj["race"] = {}
				for i in range(0, len(race_labels)):
					race_label = race_labels[i]
					race_prediction = 100 * race_predictions[i] / sum_of_predictions
					resp_obj["race"][race_label] = race_prediction
				
				resp_obj["dominant_race"] = race_labels[np.argmax(race_predictions)]

			action_idx = action_idx + 1
		
		if bulkProcess == True:
			resp_objects.append(resp_obj)
		else:
			return resp_obj

	if bulkProcess == True:
		
		resp_obj = {}
		
		for i in range(0, len(resp_objects)):
			resp_item = resp_objects[i]
			resp_obj["instance_%d" % (i+1)] = resp_item

		return resp_obj

def find(img_path, db_path, model_name ='VGG-Face', distance_metric = 'cosine', model = None, enforce_detection = True, detector_backend = 'mtcnn'):
	
	tic = time.time()
	
	img_paths, bulkProcess = initialize_input(img_path)
	functions.initialize_detector(detector_backend = detector_backend)
	
	#-------------------------------
	
	#model metric pairs for ensemble
	model_names = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']
	metric_names = ['cosine', 'euclidean', 'euclidean_l2']
	
	#-------------------------------
	
	if os.path.isdir(db_path) == True:
		
		#---------------------------------------
		
		if model == None:
			
			if model_name == 'Ensemble':
				print("Ensemble learning enabled")				
				models = Boosting.loadModel()
			
			else: #model is not ensemble
				model = build_model(model_name)
				
		else: #model != None
			print("Already built model is passed")
			
			if model_name == 'Ensemble':
				
				Boosting.validate_model(model)				
				models = model.copy()
				
		#---------------------------------------
		
		file_name = "representations_%s.pkl" % (model_name)
		file_name = file_name.replace("-", "_").lower()
		
		if path.exists(db_path+"/"+file_name):
			
			print("WARNING: Representations for images in ",db_path," folder were previously stored in ", file_name, ". If you added new instances after this file creation, then please delete this file and call find function again. It will create it again.")
			
			f = open(db_path+'/'+file_name, 'rb')
			representations = pickle.load(f)
			
			print("There are ", len(representations)," representations found in ",file_name)
			
		else:
			employees = []
			
			for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
				for file in f:
					if ('.jpg' in file):
						exact_path = r + "/" + file
						employees.append(exact_path)
			
			if len(employees) == 0:
				raise ValueError("There is no image in ", db_path," folder!")
			
			#------------------------
			#find representations for db images
			
			representations = []
			
			pbar = tqdm(range(0,len(employees)), desc='Finding representations')
			
			#for employee in employees:
			for index in pbar:
				employee = employees[index]
				
				if model_name != 'Ensemble':
					
					#input_shape = model.layers[0].input_shape[1:3] #my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
					
					input_shape = model.layers[0].input_shape
					
					if type(input_shape) == list:
						input_shape = input_shape[0][1:3]
					else:
						input_shape = input_shape[1:3]
					
					input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
					
					img = functions.preprocess_face(img = employee, target_size = (input_shape_y, input_shape_x), enforce_detection = enforce_detection, detector_backend = detector_backend)
					representation = model.predict(img)[0,:]
					
					instance = []
					instance.append(employee)
					instance.append(representation)
					
				else: #ensemble learning
					
					instance = []
					instance.append(employee)
					
					for j in model_names:
						ensemble_model = models[j]
						
						#input_shape = model.layers[0].input_shape[1:3] #my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
	
						input_shape = ensemble_model.layers[0].input_shape
						
						if type(input_shape) == list:
							input_shape = input_shape[0][1:3]
						else:
							input_shape = input_shape[1:3]
						
						input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
						
						img = functions.preprocess_face(img = employee, target_size = (input_shape_y, input_shape_x), enforce_detection = enforce_detection, detector_backend = detector_backend)
						representation = ensemble_model.predict(img)[0,:]
						instance.append(representation)
				
				#-------------------------------
				
				representations.append(instance)
			
			f = open(db_path+'/'+file_name, "wb")
			pickle.dump(representations, f)
			f.close()
			
			print("Representations stored in ",db_path,"/",file_name," file. Please delete this file when you add new identities in your database.")
		
		#----------------------------
		#we got representations for database
		
		if model_name != 'Ensemble':
			df = pd.DataFrame(representations, columns = ["identity", "representation"])
		else: #ensemble learning
			df = pd.DataFrame(representations, columns = ["identity", "VGG-Face_representation", "Facenet_representation", "OpenFace_representation", "DeepFace_representation"])
			
		df_base = df.copy()
		
		resp_obj = []
		
		global_pbar = tqdm(range(0,len(img_paths)), desc='Analyzing')
		for j in global_pbar:
			img_path = img_paths[j]
		
			#find representation for passed image
			
			if model_name == 'Ensemble':
				for j in model_names:
					ensemble_model = models[j]
					
					#input_shape = ensemble_model.layers[0].input_shape[1:3] #my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
	
					input_shape = ensemble_model.layers[0].input_shape
					
					if type(input_shape) == list:
						input_shape = input_shape[0][1:3]
					else:
						input_shape = input_shape[1:3]
					
					img = functions.preprocess_face(img = img_path, target_size = input_shape, enforce_detection = enforce_detection, detector_backend = detector_backend)
					target_representation = ensemble_model.predict(img)[0,:]
					
					for k in metric_names:
						distances = []
						for index, instance in df.iterrows():
							source_representation = instance["%s_representation" % (j)]
							
							if k == 'cosine':
								distance = dst.findCosineDistance(source_representation, target_representation)
							elif k == 'euclidean':
								distance = dst.findEuclideanDistance(source_representation, target_representation)
							elif k == 'euclidean_l2':
								distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation), dst.l2_normalize(target_representation))
							
							distances.append(distance)
						
						if j == 'OpenFace' and k == 'euclidean':
							continue
						else:
							df["%s_%s" % (j, k)] = distances
				
				#----------------------------------
				
				feature_names = []
				for j in model_names:
					for k in metric_names:
						if j == 'OpenFace' and k == 'euclidean':
							continue
						else:
							feature = '%s_%s' % (j, k)
							feature_names.append(feature)
				
				#print(df[feature_names].head())
				
				x = df[feature_names].values
				
				#----------------------------------
				#lightgbm model				
				deepface_ensemble = Boosting.build_gbm()
				
				y = deepface_ensemble.predict(x)
				
				verified_labels = []; scores = []
				for i in y:
					verified = np.argmax(i) == 1
					score = i[np.argmax(i)]
					
					verified_labels.append(verified)
					scores.append(score)
				
				df['verified'] = verified_labels
				df['score'] = scores
				
				df = df[df.verified == True]
				#df = df[df.score > 0.99] #confidence score
				df = df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)
				df = df[['identity', 'verified', 'score']]
				
				resp_obj.append(df)
				df = df_base.copy() #restore df for the next iteration
				
				#----------------------------------
			
			if model_name != 'Ensemble':
				
				#input_shape = model.layers[0].input_shape[1:3] #my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
				
				input_shape = model.layers[0].input_shape
				
				if type(input_shape) == list:
					input_shape = input_shape[0][1:3]
				else:
					input_shape = input_shape[1:3]
				
				input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
				
				#------------------------
				
				img = functions.preprocess_face(img = img_path, target_size = (input_shape_y, input_shape_x), enforce_detection = enforce_detection, detector_backend = detector_backend)
				target_representation = model.predict(img)[0,:]
		
				distances = []
				for index, instance in df.iterrows():
					source_representation = instance["representation"]
					
					if distance_metric == 'cosine':
						distance = dst.findCosineDistance(source_representation, target_representation)
					elif distance_metric == 'euclidean':
						distance = dst.findEuclideanDistance(source_representation, target_representation)
					elif distance_metric == 'euclidean_l2':
						distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation), dst.l2_normalize(target_representation))
					else:
						raise ValueError("Invalid distance_metric passed - ", distance_metric)
					
					distances.append(distance)
				
				threshold = functions.findThreshold(model_name, distance_metric)
				
				df["distance"] = distances
				df = df.drop(columns = ["representation"])
				df = df[df.distance <= threshold]
			
				df = df.sort_values(by = ["distance"], ascending=True).reset_index(drop=True)
				resp_obj.append(df)
				df = df_base.copy() #restore df for the next iteration
			
		toc = time.time()
		
		print("find function lasts ",toc-tic," seconds")
		
		if len(resp_obj) == 1:
			return resp_obj[0]
		
		return resp_obj
		
	else:
		raise ValueError("Passed db_path does not exist!")
		
	return None
		
def stream(db_path = '', model_name ='VGG-Face', distance_metric = 'cosine', enable_face_analysis = True):
	
	functions.initialize_detector(detector_backend = 'opencv')
	
	realtime.analysis(db_path, model_name, distance_metric, enable_face_analysis)

def detectFace(img_path, detector_backend = 'mtcnn'):
	
	functions.initialize_detector(detector_backend = detector_backend)
	
	img = functions.preprocess_face(img = img_path, detector_backend = detector_backend)[0] #preprocess_face returns (1, 224, 224, 3)
	return img[:, :, ::-1] #bgr to rgb

def initialize_input(img1_path, img2_path = None):
	
	"""
	verify, analyze and find functions build complex machine learning models in every call.
	To avoid memory problems, you can pass image pairs as array.
	
	This function manages this usage is enabled or not
	
	E.g.
	result  = DeepFace.verify("img1.jpg", "img2.jpg")
	results = DeepFace.verify([['img1.jpg', 'img2.jpg'], ['img1.jpg', 'img3.jpg']])
	"""

	if type(img1_path) == list:
		bulkProcess = True
		img_list = img1_path.copy()
	else:
		bulkProcess = False
		if img2_path != None:
			img_list = [[img1_path, img2_path]]
		else:
			img_list = [img1_path]
	
	return img_list, bulkProcess
	
#---------------------------
#main

functions.initializeFolder()
