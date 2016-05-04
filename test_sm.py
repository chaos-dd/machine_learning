# coding=utf-8

from classification import SoftMax
from utils import svm_read_problem
import evaluation.classifier_eval as ev

x, y = svm_read_problem("D:\\train_data\\data_tiny.libsvm")

print('x.shape:', x.shape, 'y.shape:', y.shape)
classifier = SoftMax(lam=0.1, normalize=True, debug=False)

classifier.train(x, y)
predict_y = classifier.predict(x)
err_list = predict_y == y

corr = 0
for err in err_list:
    if err:
        corr += 1
print("precision:", corr / len(err_list))

ev.eval(classifier, x, y, debug=False)
