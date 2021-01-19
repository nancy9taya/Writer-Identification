# from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


votingClf=None

def training(x_train,y_train):
    global votingClf

    votingClf = VotingClassifier([('clf1',SVC(probability=True)),('clf2',DecisionTreeClassifier())],voting='soft') #
    learnRate=1.5
    numClassifiers=50;
    votingClf = AdaBoostClassifier(base_estimator = votingClf,  n_estimators=numClassifiers,
                                 learning_rate=learnRate,
                                 algorithm="SAMME")
    votingClf.fit(x_train,y_train)
    return


def predict_clf(Y_test, X_test):
    global votingClf
    prediction = votingClf.predict(X_test)
    print("Actual Classes:", Y_test)
    print("Prediction:", prediction)
    accuracy = sum(Y_test == prediction) / len(Y_test)
    print("The classification accuracy: " + str(accuracy))
    if (accuracy < 1):
        exit()
    return
