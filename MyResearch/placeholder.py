from new_dataloader import Dataset

 # Create dataloader
dataset = Dataset(data_dir='Data/final_dataset/',
                    batch_size=2,
                    )

train_loader = dataset.get_train_loader()


for epoch in range(10):
    for i, data in enumerate(train_loader):
        input = data[0]
        target = data[1]

    dataset.reset()
