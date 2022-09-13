import sklearn 
from sklearn.linear_model import LogisticRegression 

class Model():
    def __init__(self, type, random_state=42, kernel= 'rbf', max_depth = 10, wandb=None, num_estimator= 100, voting="hard", save_model= None):
        self.type = type
        self.random_state = random_state
        self.kernel = kernel
        self.max_depth = max_depth
        self.num_estimator = num_estimator
        self.voting = voting
        self.save_model = save_model
        self.model = None
    def define_type(self):
        model = None
        if self.type == 'Logistic regression':
            model = LogisticRegression(self.random_state)
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
    def fit(self, X_train, y_train):
        self.model = define_type(self)
        self.model.fit(X_train, y_train)
        return self.model
    def score(self, X, y):
        return self.model.score(X,y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X)
    
    def Visualize_metrics(self, X, y):
        y_pred = self.predict(X)
        