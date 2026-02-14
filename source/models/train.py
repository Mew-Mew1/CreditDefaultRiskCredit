import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
import xgboost as xgb

from source.features.preprocess import build_preprocessor
from source.utils.logger import get_logger

logger = get_logger(__name__)

def load_data(filepath):
    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        X = df.drop(['default payment next month', 'ID'], axis=1)
        y = df['default payment next month']
        logger.info(f"Data loaded successfully. Shape: X={X.shape}, y={y.shape}")
        return X, y
    except FileNotFoundError:
        logger.error(f"Data file not found at {filepath}. Please ensure the data is downloaded and processed.")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        raise

def train_models():
    try:
        logger.info("Starting model training pipeline...")
        X, y = load_data('data/raw_csv/default_of_credit_card_clients.csv')
        
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        preprocessor = build_preprocessor()

        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }

        param_grids = {
            'LogisticRegression': {'classifier__C': [0.01, 0.1, 1, 10]},
            'RandomForest': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None]
            },
            'XGBoost': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            }
        }

        best_model = None
        best_score = 0
        best_model_name = ""

        logger.info("Starting hyperparameter tuning...")
        for name, model in models.items():
            logger.info(f"Tuning {name}...")
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', model)])
            
            search = RandomizedSearchCV(pipeline, param_distributions=param_grids[name], 
                                        n_iter=5, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
            search.fit(X_train, y_train)
            
            logger.info(f"{name} Best ROC-AUC: {search.best_score_:.4f}")
            
            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_model = search.best_estimator_
                best_model_name = name

        logger.info(f"Hyperparameter tuning complete. Best Model: {best_model_name} with ROC-AUC: {best_score:.4f}")
        
        logger.info("Evaluating best model on test set...")
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        test_roc_auc = roc_auc_score(y_test, y_proba)
        logger.info(f"Test Set ROC-AUC: {test_roc_auc:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

        logger.info("Saving best model artifact...")
        os.makedirs('models', exist_ok=True)
        models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, 'best_model_pipeline.pkl')
        joblib.dump(best_model, model_path)
        logger.info(f"Model successfully saved to {model_path}")

    except Exception as e:
        logger.critical(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train_models()