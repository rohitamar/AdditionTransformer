import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from TransformerModel import TransformerModel
from AdditionDataset import AdditionDataset

model = TransformerModel(15, 512)
model = model.cuda()

train_dataset = AdditionDataset(5, 4096)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

epochs = 8000
initial_lr = 1e-4
peak_lr = 1e-3
total_steps = epochs * len(train_dataloader)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=peak_lr, 
                                                total_steps=total_steps, 
                                                pct_start=0.3, 
                                                anneal_strategy='cos')

writer = SummaryWriter()

for epoch in range(epochs):
    correct = total = 0
    tot_loss = 0.0

    if epoch % 10 == 0:
        train_dataset = AdditionDataset(5, 4096)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for batch in train_dataloader:
        input, target = batch
        input, target = input.cuda(), target.cuda()

        output = model(input)

        predictions = torch.argmax(output, dim=-1)
        total += input.shape[0]
        check = torch.all(predictions == target, dim=-1)
        correct += check.sum()

        output = output.view(-1, 15)
        target = target.flatten()

        loss = criterion(output, target)
        tot_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.add_scalar("Loss", tot_loss, epoch)
    writer.add_scalar("Accuracy", correct / total, epoch)

    print(f"Epoch {epoch} Accuracy {correct} / {total} Loss {tot_loss}")

writer.close()