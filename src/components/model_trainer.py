
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

class ModelTrainer:

    def TrainModel(self, x, y, preprocessor):

        # train test split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        # full pipeline (preprocessing + model)
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model",RandomForestClassifier(class_weight='balanced',random_state=42))
            ]
        )

        # train model
        model_pipeline.fit(x_train, y_train)

        # prediction
        y_pred = model_pipeline.predict(x_test)

        # accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print("Model Accuracy:", accuracy)
        fold=KFold(n_splits=5,shuffle=True,random_state=42)
        cross=cross_val_score(model_pipeline,x,y,cv=fold)

        print("cross_validation_score:",cross.mean())

        params={
            'model__n_estimators':[200,300,400],
            'model__criterion':['gini','entropy'],
            'model__max_depth':[3,4,5,6],
            'model__min_samples_split':[4,5,6],
            'model__min_samples_leaf':[3,4,5,6]
            }
        grid=GridSearchCV(model_pipeline,params,cv=5)
        grid.fit(x_train,y_train)
        print("best_grid_parameters",grid.best_params_)
        print("best_grid_score",grid.best_score_)

        print("Improved Accuracy :",accuracy_score(y_test,y_pred))

        # create models folder
        os.makedirs("models", exist_ok=True)

        # save model
        joblib.dump(model_pipeline, "models/model.pkl")

        print("Model saved successfully!")

        return model_pipeline
