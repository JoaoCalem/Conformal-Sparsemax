from conformal_sparsemax.ConformalPredictor import ConformalPredictor,SparseScore,SoftmaxScore

# Get non-conformity scores
predictions_path = ''
dataset = ...
pred_cal_path = 'predictions/CIFAR100_cal_NLLLoss_softmax_proba.pickle'
pred_test_path = 'predictions/CIFAR100_test_NLLLoss_softmax_proba.pickle'
true_cal_path = 'predictions/CIFAR100_cal_true.pickle'
true_test_path = 'predictions/CIFAR100_test_true.pickle'
pred_cal, pred_test, true_cal, true_test = get_data(pred_cal_path, pred_test_path,true_cal_path, true_test_path)