import sklearn 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

class Model():
    def __init__(self, type, random_state=42, kernel= 'rbf', max_depth = 10, wandb=None, num_estimator= 100, voting="hard", save_model= None):
        self.type = type
        self.random_state = random_state
        self.kernel = kernel
        self.max_depth = max_depth
        self.num_estimator = num_estimator
        self.voting = voting
    def get_model(self):
        model = None
        if self.type == 'Logistic regression':
            model = LogisticRegression(random_state = self.random_state)
        elif self.type == 'SVM':
            model = SVC(kernel = self.kernel, random_state = self.random_state)
        elif self.type == 'Decision Trees':
            model = DecisionTreeClassifier(max_depth = 10)
        elif self.type == 'Random Forest':
            model = RandomForestClassifier(n_estimators=self.num_estimator, random_state= self.random_state)
        elif self.type == "Voting Classifier":
            model = VotingClassifier(estimators=[
                        ('LGR', LG_classifier), ('SVC', classifier), ('Decision_Tree', tree_clf), ('RandomForest', RFC)],
                        voting= self.voting
                        )
        return model