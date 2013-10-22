import math, random

# SAmple confidence vectors 
# predicted_confidence_vector_a = 	[0,	0,	0.6,	0,	0, .4,	0,	1,	0,	0,	0,	0,	0,	0, 0,	0,	0,	0,	0,	0,	1,	0,	0,	0]
# predicted_confidence_vector_b = 	[0,	0,	0.6,	0,	0, .4,	0,	1,	0,	0,	0,	0,	0,	0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
# predictions_list = [predicted_confidence_vector_a, predicted_confidence_vector_b]
# gold_confidence_vector_a =	 		[0,	0,	0.5,	0,	0, .5,	0,	1,	0,	0,	0,	0,	0,	0, 0,	0,	0,	0,	0,	0,	1,	0,	0,	0]
# gold_confidence_vector_b =	 		[0,	0,	0.6,	0,	0, .4,	0,	1,	0,	0,	0,	0,	0,	0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
# gold_list = [gold_confidence_vector_a, gold_confidence_vector_b]
# Note: These only differ in confidence of s2 and s5 in vector a
# These differences make the squared error 0.1^2 + 0.1^2 = 0.02
# We expect root mean squared error to be math.sqrt(0.02 / 2 * 24) = 0.020412414523193152

# print "predicted_confidence_vector is "
# for prediction in predictions_list:
# 	print prediction
# print "gold_confidence_vector is "
# for gold in gold_list:
# 	print gold

# def single_data_point_mse(predicted_confidence_vector, gold_confidence_vector):
# 	numLabels = len(predicted_confidence_vector)
# 	squared_error = 0.0
# 	for label_index in range(numLabels):
# 		squared_error += math.pow((predicted_confidence_vector[label_index] - gold_confidence_vector[label_index]), 2)
# 	mean_squared_error = squared_error / numLabels
# 	return mean_squared_error

def single_data_point_se(predicted_confidence_vector, gold_confidence_vector):
	numLabels = len(predicted_confidence_vector)
	squared_error = 0.0
	for label_index in range(numLabels):
		squared_error += math.pow((predicted_confidence_vector[label_index] - gold_confidence_vector[label_index]), 2)
	return squared_error

def rmse(predictions_list, gold_list):
	numPredictions = len(predictions_list)
	if numPredictions <= 0: 
		print "numPredictions <= 0"
	if len(gold_list) <= 0:
		print "len(gold_list) <= 0"
	if numPredictions != len(gold_list):
		print "predictions_list and gold_list do not match"

	numLabels = len(predictions_list[0])
	if numLabels <= 0:
		print "prediction numLabels <= 0"
	if len(gold_list[0]) <= 0:
		print "gold numLabels <= 0"
	if numLabels != len(gold_list[0]):
		print "length of a prediction label vector and gold label vector do not match"

	print "numPredictions is %d\nnumLabels is %d" %(numPredictions, numLabels)

	# Note: ensure that numPredictions = len(gold_list) 
	# Ensure that numPredictions > 0 and len(gold_list) > 0
	# numLabels = len(gold_list[0])

	total_squared_error = 0.0

	for prediction_index in range(len(predictions_list)):
		# total_mean_squared_error += single_data_point_mse(predictions_list[prediction_index], gold_list[prediction_index])
		total_squared_error += single_data_point_se(predictions_list[prediction_index], gold_list[prediction_index])
	total_mean_squared_error = total_squared_error / (numPredictions * numLabels)
	root_mean_squared_error = math.sqrt(total_mean_squared_error)
	return root_mean_squared_error

#print "rmse is %f" %(rmse(predictions_list, gold_list))


# Note: can copy or use scikit's: http://scikit-learn.org/stable/modules/cross_validation.html

# def k_folds_validation(dataList):
# 	numDataPoints = len(dataList)
# 	numDataPointsPerFold = numDataPoints / 5 # Note: we can lose up to 4 total data numDataPoints