import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from scipy.sparse import load_npz
import mlflow
import mlflow.keras

# ===============================
# 1. CONFIGURAR MLflow TRACKING SERVER
# ===============================
mlflow.set_tracking_uri("http://ec2-34-201-213-246.compute-1.amazonaws.com:8080")  # Cambia por tu URL real
experiment_name = "tf-idf_ngram_1,2_nueral_network_remove_punctutation_true"
mlflow.set_experiment(experiment_name)

# ===============================
# 2. FUNCIONES AUXILIARES
# ===============================
def sparse_to_sparse_tensor_batch(X_sparse):
    X_coo = X_sparse.tocoo()
    indices = np.mat([X_coo.row, X_coo.col]).transpose()
    values = X_coo.data.astype(np.float32)
    shape = X_sparse.shape
    return tf.sparse.SparseTensor(indices, values, shape)

def sparse_generator(X_sparse, y, batch_size, steps_per_epoch):
    n_samples = X_sparse.shape[0]
    for step in range(steps_per_epoch):
        start = (step * batch_size) % n_samples
        end = min(start + batch_size, n_samples)
        X_batch = X_sparse[start:end]
        y_batch = y[start:end].astype(np.float32)
        yield sparse_to_sparse_tensor_batch(X_batch), y_batch

def predict_batches(model, X_sparse, batch_size):
    n_samples = X_sparse.shape[0]
    y_preds = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        X_batch = sparse_to_sparse_tensor_batch(X_sparse[start:end])
        y_batch_pred = model(X_batch, training=False)
        y_preds.append(y_batch_pred.numpy())
    return (np.vstack(y_preds) > 0.5).astype(int)

# ===============================
# 3. CALLBACK F1 SCORE POR EPOCA
# ===============================
class F1Callback(Callback):
    def __init__(self, X_val, y_val, batch_size=512):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        y_pred = predict_batches(self.model, self.X_val, self.batch_size)
        f1 = f1_score(self.y_val, y_pred)
        print(f"\nEpoch {epoch+1} — Dev F1 Score: {f1:.4f}")
        if logs is not None:
            logs["val_f1"] = f1  # Para MLflow

# ===============================
# 4. CARGAR DATOS
# ===============================
X_train = np.load("X_train.npy",allow_pickle=True).item()
y_train = np.load("y_train.npy")
print(type(X_train))
X_dev = np.load("X_dev.npy",allow_pickle=True).item()
y_dev = np.load("y_dev.npy")

X_test = np.load("X_test.npy",allow_pickle=True).item()
y_test = np.load("y_test.npy")

batch_size = 512
epochs = 15
n_splits = 3

# ===============================
# 5. CROSS-VALIDATION
# ===============================
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
X_train_array = np.arange(X_train.shape[0])

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_array)):
    print(f"\n=== Fold {fold+1}/{n_splits} ===")

    X_tr_fold = X_train[train_idx]
    y_tr_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # ===============================
    # 6. DEFINIR MODELO DENSO
    # ===============================
    inputs = Input(shape=(X_train.shape[1],), sparse=True)
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    f1_cb = F1Callback(X_val_fold, y_val_fold, batch_size=batch_size)

    # ===============================
    # 7. ENTRENAR MODELO
    # ===============================
    steps_per_epoch = int(np.ceil(len(train_idx)/batch_size))
    validation_steps = int(np.ceil(len(val_idx)/batch_size))

    with mlflow.start_run(run_name=f"Fold_{fold+1}"):
        history = model.fit(
            sparse_generator(X_tr_fold, y_tr_fold, batch_size, steps_per_epoch),
            validation_data=sparse_generator(X_val_fold, y_val_fold, batch_size, validation_steps),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[early_stop, f1_cb],
            verbose=1
        )

        # ===============================
        # 8. EVALUACIÓN FINAL EN DEV
        # ===============================
        y_dev_pred = predict_batches(model, X_dev, batch_size)
        f1_dev = f1_score(y_dev, y_dev_pred)
        print(f"Final Dev F1 Score (Fold {fold+1}): {f1_dev:.4f}")

        # ===============================
        # 9. REGISTRAR EN MLflow
        # ===============================
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("fold", fold+1)
        mlflow.log_metric("dev_f1_score", f1_dev)
        mlflow.keras.log_model(model, artifact_path=f"model_fold_{fold+1}")

# ===============================
# 10. EVALUAR FINAL EN TEST
# ===============================
y_test_pred = predict_batches(model, X_test, batch_size)
f1_test = f1_score(y_test, y_test_pred)
print(f"\nTest F1 Score: {f1_test:.4f}")
