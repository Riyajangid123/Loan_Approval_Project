from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer

class TrainPipeline:

    def run_pipeline(self):

        # Step 1 : Data Ingestion
        ingestion = DataIngestion()
        data = ingestion.Ingest_data(r"C:\Users\DELL\OneDrive\Desktop\Loan_prediction\data\data.csv")

        print("Data Ingestion Completed")

        # Step 2 : Data Preprocessing
        preprocessing = DataPreprocessing()
        x, y, preprocessor = preprocessing.PreProcessData(data)

        print("Data Preprocessing Completed")

        # Step 3 : Model Training
        trainer = ModelTrainer()
        trainer.TrainModel(x, y, preprocessor)

        print("Model Training Completed")


if __name__ == "__main__":

    pipeline = TrainPipeline()
    pipeline.run_pipeline()
