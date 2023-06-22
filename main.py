import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.nn.functional as F



# Set the seed for reproducibility
random.seed(42)
torch.manual_seed(42)

class GuitarSoundDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        # Asegurarse de que el waveform tenga una longitud de 4410
        if waveform.shape[1] < 63*100:
            num_repeats = 63*100 // waveform.shape[1] + 1
            repeated_waveform = waveform.repeat(1, num_repeats)
            waveform = repeated_waveform[:, :63*100]
        else:
            waveform = waveform[:, :63*100]

        # Compute the STFT magnitude spectrogram
        n_fft = 126
        win_length = None
        hop_length = 100

        # Define transform
        spectrogram = torch.stft(
            input=waveform,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            return_complex=False,
            normalized=True
        )

        # Compute Mel Spectrogram
        mel_spectrogram_transform = transforms.MelSpectrogram(sample_rate, 
                                                            n_fft=n_fft, 
                                                            hop_length=hop_length,
                                                            normalized=True,
                                                            pad_mode="reflect")
        mel_spectrogram = mel_spectrogram_transform(waveform)

        # Compute MFCC
        mfcc_transform = transforms.MFCC(sample_rate, n_mfcc=64, melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'win_length': win_length})
        mfcc = mfcc_transform(waveform)

        # Stack the features into an image-like representation
        spectrogram_amplitude = torch.abs(spectrogram[0, :, :, 0])
        spectrogram_db = transforms.AmplitudeToDB()(spectrogram_amplitude)
        mel_spectrogram_db = transforms.AmplitudeToDB()(mel_spectrogram)
        mel_spectrogram_db = F.interpolate(mel_spectrogram_db.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)
        mfcc_db = transforms.AmplitudeToDB()(mfcc)[0]

        image1=torch.transpose(spectrogram_db, 0, 1)
        image2=mel_spectrogram_db[0]
        image3=mfcc_db

        # Combine the three channels into an RGB image
        image = torch.stack([image1, image2, image3], dim=0).detach().numpy()

        # Determine the label based on the file path
        label = 0 if 'acoustic' in file_path else 1

        return image, label

# Define the file paths for the acoustic and electric notes
acoustic_note_paths = ['C:/Users/usuario/proyect/Data/acoustic_note_{}.wav'.format(n) for n in range(1, 6300)]
electric_note_paths = ['C:/Users/usuario/proyect/Data/note_{}.wav'.format(n) for n in range(1, 6300)]

# Combine the file paths
file_paths = acoustic_note_paths + electric_note_paths

# Split the dataset into training and validation sets
train_files, valid_files = train_test_split(file_paths, test_size=0.2, random_state=42)

# Create the dataset for training and validation sets
train_dataset = GuitarSoundDataset(train_files)
valid_dataset = GuitarSoundDataset(valid_files)

# Create data loaders for training and validation sets
batch_size = 10
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

class GuitarSoundModel(nn.Module):
    def __init__(self, input_size):
        super(GuitarSoundModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.001)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.leakyrelu4 = nn.LeakyReLU(negative_slope=0.001)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calcula la salida de la capa pool4
        self.output_size = input_size // 4
        
        self.fc1 = nn.Linear(128 * self.output_size * self.output_size, 32)
        self.leakyrelu5 = nn.LeakyReLU(negative_slope=0.001)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.leakyrelu4(x)
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.leakyrelu5(x)
        x = self.fc2(x)
        
        return x


# Initialize the model, loss function, and optimizer
input_size = 64
model = GuitarSoundModel(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for image, labels in train_dataloader:
        optimizer.zero_grad()
        inputs = image 

        if len(inputs) < batch_size:
            break

        # Resto del código
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  
        for image, labels in valid_dataloader:
            inputs = image
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_loss /= len(valid_dataloader)
    valid_accuracy = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2%}")

