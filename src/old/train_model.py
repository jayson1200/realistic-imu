import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import re
import pickle

from dataset import MotionDataset
from time_model import TimeModel

if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

torch.set_float32_matmul_precision('high')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 8000
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0
data_path = "/home/meribejayson/Desktop/Projects/realistic-imu/data/total_capture_data"
base_path = "/home/meribejayson/Desktop/Projects/realistic-imu/models"
D_MODEL = 512
INPUT_EMBEDDING_DIM = 42
NUM_ENCODERS = 12

model = TimeModel(d_model=D_MODEL,
                  inp_emb_dim=INPUT_EMBEDDING_DIM,
                  device=device,
                  num_encoders=NUM_ENCODERS).to(device)

model = torch.compile(model)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

csv_file = os.path.join(data_path, "training_log.csv")

# Ensure data folder exists
os.makedirs(data_path, exist_ok=True)

# Initialize the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Test Loss"])  # Header row

train_subjects = ["s1", "s2", "s3", "s4"]
test_subjects = ["s5"]

train_dataset = MotionDataset(data_path, subjects=train_subjects, dataset_type="train")
test_dataset = MotionDataset(data_path, subjects=test_subjects, dataset_type="test")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

criterion = nn.SmoothL1Loss()

model.to(device)

train_losses = []
test_losses = []
epochs_list = [-1]
test_epochs = [-1]

# Log initial loss without training
print("Evaluating initial loss before training...")
model.eval()
initial_train_loss = 0
initial_test_loss = 0

with torch.no_grad():
    for mocap_data, imu_data in train_loader:
        mocap_data = mocap_data.to(device)
        imu_data = imu_data.to(device)

        if mocap_data.shape[1] != imu_data.shape[1]:
            mocap_data = mocap_data[:, 1:-1, :]

        min_orig_accel_norm = mocap_data[0, :, 0:26:2].T
        output = model(mocap_data, min_orig_accel_norm)
        loss = criterion(output, imu_data.squeeze(0).T)
        initial_train_loss += loss.item()
    initial_train_loss /= len(train_loader)

    for mocap_data, imu_data in test_loader:
        mocap_data = mocap_data.to(device)
        imu_data = imu_data.to(device)

        if mocap_data.shape[1] != imu_data.shape[1]:
            mocap_data = mocap_data[:, 1:-1, :]

        min_orig_accel_norm = mocap_data[0, :, 0:26:2].T
        output = model(mocap_data, min_orig_accel_norm)
        loss = criterion(output, imu_data.squeeze(0).T)
        initial_test_loss += loss.item()
    initial_test_loss /= len(test_loader)

train_losses.append(initial_train_loss)
test_losses.append(initial_test_loss)

progress_bar = tqdm(range(EPOCHS), desc="Training Progress", position=0, leave=True)

for epoch in progress_bar:
    model.train()
    epoch_train_loss = 0

    for mocap_data, imu_data in train_loader:
        optimizer.zero_grad()
        mocap_data = mocap_data.to(device)
        imu_data = imu_data.to(device)

        if mocap_data.shape[1] != imu_data.shape[1]:
            mocap_data = mocap_data[:, 1:-1, :]

        min_orig_accel_norm = mocap_data[0, :, 0:26:2].T

        output = model(mocap_data, min_orig_accel_norm)
        loss = criterion(output, imu_data.squeeze(0).T)
        epoch_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    scheduler.step()

    train_loss = epoch_train_loss / len(train_loader)
    epochs_list.append(epoch + 1)
    train_losses.append(train_loss)

    if (epoch + 1) % 10 == 0:
        model.eval()
        epoch_test_loss = 0

        with torch.no_grad():
            for mocap_data, imu_data in test_loader:
                mocap_data = mocap_data.to(device)
                imu_data = imu_data.to(device)

                if mocap_data.shape[1] != imu_data.shape[1]:
                    mocap_data = mocap_data[:, 1:-1, :]

                min_orig_accel_norm = mocap_data[0, :, 0:26:2].T

                output = model(mocap_data, min_orig_accel_norm)
                loss = criterion(output, imu_data.squeeze(0).T)
                epoch_test_loss += loss.item()

        test_loss = epoch_test_loss / len(test_loader)
        test_epochs.append(epoch + 1)
        test_losses.append(test_loss)
    else:
        test_loss = None

    # Save to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss, test_loss if test_loss is not None else ""])

    # Log the current learning rate
    current_lr = optimizer.param_groups[0]['lr']

    progress_desc = f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | LR: {current_lr:.12f}"
    if test_loss is not None:
        progress_desc += f" | Test Loss: {test_loss:.4f}"
    progress_bar.set_description(progress_desc)

os.makedirs(base_path, exist_ok=True)
model_files = [f for f in os.listdir(base_path) if re.match(r"model_\d+\.pkl", f)]

if model is not None:
    if model_files:
        max_num = max(int(re.search(r"model_(\d+)\.pkl", f).group(1)) for f in model_files)
    else:
        max_num = 0
    new_model_name = f"model_{max_num + 1}.pkl"
    save_path = os.path.join(base_path, new_model_name)
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to: {save_path}")
else:
    if not model_files:
        raise FileNotFoundError("No model files found in the directory.")
    latest_model_file = max(model_files, key=lambda f: int(re.search(r"model_(\d+)\.pkl", f).group(1)))
    load_path = os.path.join(base_path, latest_model_file)
    with open(load_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Loaded model from: {load_path}")




plt.figure(figsize=(12.8, 9.6))

# Plotting the losses
plt.plot(epochs_list, train_losses, label='Train Loss', color='blue')
plt.plot(test_epochs, test_losses, label='Test Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.show()


print(f"Training completed! Logs saved to {csv_file}")
