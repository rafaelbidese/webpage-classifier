from fetch_dataset import fetch_train_data
from fetch_dataset import fetch_test_data
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


print("Fetching data...")

data_train = fetch_train_data()
data_test = fetch_test_data()

print("TfidfVectorizing data...")

from sklearn.externals import joblib

#vectorizer = joblib.load('vectorizer.pkl')
#X_train = vectorizer.transform(data_train.data)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

joblib.dump(vectorizer, 'vectorizer.pkl') 

y_train, y_test = data_train.target, data_test.target

target_names = data_train.target_names

joblib.dump(target_names, 'target_names.pkl') 


clf1 = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=50, n_iter=None,
       n_jobs=1, penalty='l1', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)

clf2 = RandomForestClassifier(n_estimators=100, class_weight="balanced", oob_score=True)

clf3 = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l1', random_state=None, tol=0.001,
     verbose=0)

xgb = XGBClassifier()
	 
models = [clf1, clf2, clf3, xgb]

print("Start fitting the models...")

out_of_sample = []
for model in models:
	label = model.__class__.__name__
	
	#xscores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=3)
	#print("3-fold cross-validation accuracy:   %0.3f +/- %0.2f [%s]" % (xscores.mean(), xscores.std(),label))
	
	model.fit(X_train, y_train)
	joblib.dump(model, label+'.pkl')
	
	pred = model.predict(X_test)
	score = metrics.accuracy_score(y_test, pred)

	print("out-of-sample accuracy:   %0.3f [%s]" % (score,label))
	out_of_sample.append((label,score))
	
	print("classification report:")
	print(metrics.classification_report(y_test, pred, target_names=target_names))

print("Models trained!")
	
out_df = pd.DataFrame(out_of_sample, columns=['model_name', 'accuracy'])
out_df = out_df.sort_values(by='accuracy')
plt.stem(out_df['model_name'], out_df['accuracy'])
plt.title('Out-of-sample accuracy')
plt.ylabel('accuracy')
plt.show()