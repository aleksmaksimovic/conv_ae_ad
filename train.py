from model import ConvAutoEncoder
from data_loader import KolektorSDD2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import stats


def calculate_mean_std(dataloader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in dataloader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean, std








device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=ConvAutoEncoder()
model.to(device)



transform = transforms.Compose([
    transforms.Resize((128, 628)),
    transforms.ToTensor(),
])


 
# Load dataset
train_dataset=KolektorSDD2('/home/aleks/Documents/conv_ae_ad/dataset/train', img_transform=transform, train=True)



eval_dataset=KolektorSDD2('/home/aleks/Documents/conv_ae_ad/dataset/test', img_transform=transform, train=True)


test_dataset=KolektorSDD2('/home/aleks/Documents/conv_ae_ad/dataset/test', img_transform=transform, train=False)

#Let's use 30 % of the (negative) test data to determine the threshhold for normal instances
total_samples = len(eval_dataset)
twenty_percent_index = int(0.3 * total_samples)
subset_eval_dataset = torch.utils.data.Subset(eval_dataset, list(range(twenty_percent_index)))



# Calculate mean and std


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=128, 
                                           shuffle=True)



threshold_loader = torch.utils.data.DataLoader(dataset=subset_eval_dataset, 
                                           batch_size=1, 
                                           shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=1, 
                                           shuffle=True)

mean, std = calculate_mean_std(train_loader)

transform_with_normalization = transforms.Compose([
    transforms.Resize((128, 628)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])


transform_with_auto_augment = transforms.Compose([
    transforms.Resize((128, 628)),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])



train_dataset.img_transform = transform_with_auto_augment

eval_dataset.img_transform=transform_with_normalization

test_dataset.img_transform=transform_with_normalization

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training...", flush=True)
 
# Train the autoencoder
num_epochs = 1000
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
    
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()), flush=True)
 
# Save the model
torch.save(model.state_dict(), 'conv_autoencoder.pth')


model.load_state_dict(torch.load('conv_autoencoder.pth'))
model.eval()  # Set the model to evaluation mode


# Train the autoencoder

losses=[]
for data in threshold_loader:
    img, _ = data
    img = img.to(device)
    output = model(img)
    loss = criterion(output, img)
    
    losses.append(loss.item())
    
    
avg_losses_normal_instances=np.mean(np.array(losses))

threshold=avg_losses_normal_instances




test_losses=[]

for data in test_loader:
    img, label = data
    img = img.to(device)
    output = model(img)
    loss = criterion(output, img)    
    test_losses.append(loss.item())
    
 
# Assuming losses is a list of loss values
losses = np.array(test_losses)

results = {}

# Iterate over percentile values from 90 to 98 in 0.5 steps
for percentile_value in np.arange(90, 99.5, 0.5):
    # Step 1: Calculate the threshold for the current percentile
    threshold_value = np.percentile(losses, percentile_value)
    
    # Lists to store the ground truth labels and the predicted labels
    true_labels = []
    predicted_labels = []

    for data in test_loader:
        img, label = data
        img = img.to(device)
        output = model(img)
        loss = criterion(output, img)    

        # Step 2: Classify based on the current threshold
        if loss.item() > threshold_value:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

        true_labels.append(label.item())

    # Step 3: Calculate precision, recall, and F1 score
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    # Store the results for the current percentile
    results[percentile_value] = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'threshold_value': threshold_value
    }

# Print results for each percentile
for percentile_value, metrics in results.items():
    print(f"Percentile: {percentile_value:.1f}%", flush=True)
    print(f"  Threshold value: {metrics['threshold_value']:.4f}", flush=True)
    print(f"  Precision: {metrics['precision']:.4f}", flush=True)
    print(f"  Recall: {metrics['recall']:.4f}", flush=True)
    print(f"  F1 Score: {metrics['f1_score']:.4f}\n", flush=True)

    
    









