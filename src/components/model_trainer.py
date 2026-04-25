
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV


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
                ("model",LogisticRegression(class_weight='balanced',max_iter=1000))
            ]
        )
        param_grid = {
             'model__C':[0.01,0.2,0.3,10,0.7,0.8,2,3],
             }
        grid = GridSearchCV(model_pipeline,param_grid,cv=5)
        # train model
        grid.fit(x_train,y_train)
        print("Best Params:",grid.best_params_)
        best_model=grid.best_estimator_

        # prediction
        y_pred = best_model.predict(x_test)

        # accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print("Model Accuracy:", accuracy)
        fold=KFold(n_splits=5,shuffle=True,random_state=42)
        cross=cross_val_score(best_model,x,y,cv=fold)

        print("cross_validation_score:",cross.mean())



        # create models folder
        os.makedirs("models", exist_ok=True)

        # save model
        joblib.dump(best_model, "models/model.pkl")

        print("Model saved successfully!")

        return best_model
