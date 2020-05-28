import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unet3d import UNet3D
from dataset import Dataset
from finetune_config import FineTuneConfig
from stats import Statistics

config = FineTuneConfig()
stats = Statistics(config.epochs, config.batch_size, save_directory=config.stats_dir)
dataset = Dataset(data_dir=config.data_dir)

stats.initial_lr = config.lr
stats.tr_val_ts_split = dataset.tr_val_ts_split
stats.task = config.finetune_task

# prepare the 3D model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet3D()

#Load pre-trained weights
weight_dir = config.weights
checkpoint = torch.load(weight_dir, map_location=device)
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
model.load_state_dict(unParalled_state_dict)

model.to(device)
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])

criterion = nn.MSELoss()

if config.optimizer == "sgd":
	optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
elif config.optimizer == "adam":
	optimizer = torch.optim.Adam(model.parameters(), config.lr)
else:
	raise Exception

#lambda_lr = lambda epoch: LR_DECAY**epoch
#scheduler_actor = torch.optim.lr_scheduler.LambdaLR(optimizer_actor, lr_lambda=lambda_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.patience * 0.8), gamma=0.5)

# train the model
if config.self_supervised:
    raise NotImplementedError

best_loss = 10000000000
num_epoch_no_improvement = 0

for epoch in range(config.nb_epoch):
    stats.training_losses = [] 
    stats.validation_losses = []
    scheduler.step(epoch)
    model.train()
    x, iteration = 0 , 0
    while x != None: # go through all examples
        x ,y = dataset.get_train(config.batch_size)
        x, y = x.float().to(device), y.float().to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stats.training_losses.append(loss.item())
        training_losses_epoch_specific.append(loss.item())

        if (iteration + 1) % 5 == 0:
            print('Epoch [{}/{}], iteration {}, Loss: {:.6f}'.format(epoch + 1, config.nb_epoch, iteration + 1, np.average(stats.training_losses_epoch_specific)))
            sys.stdout.flush()
        iteration += 1

    with torch.no_grad():
        model.eval()
        x = 0
        while x != None:
            x, y = dataset.get_val(conf.batch_size)
            x, y = x.float().to(device), y.float().to(device)
            pred = model(x)
            loss = criterion(pred,y)
            stats.validation_losses.append(loss.item())
            training_losses_epoch_specific.append(loss.item())
        
	#logging
    stats.avg_training_loss_per_epoch.append(np.average(stats.training_losses))
    stats.avg_validation_loss_per_epoch.append(np.average(stats.validation_losses))
    stats.iterations.append(iteration)
    
    avg_training_loss_of_epoch = np.average(stats.training_losses)

	print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss))
	if avg_validation_loss_of_epoch < best_loss:
		print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
		best_loss = avg_validation_loss_of_epoch
		num_epoch_no_improvement = 0
		#save model
		torch.save({
			'epoch': epoch+1,
			'state_dict' : model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict()
		},os.path.join(conf.model_path_save, conf.finetune_task + ".pt"))
		print("Saving model ",os.path.join(conf.model_path_save, conf.finetune_task + ".pt"))
	else:
		print("Validation loss did not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
		num_epoch_no_improvement += 1
	if num_epoch_no_improvement >= conf.patience:
		print("Early Stopping")
		stats.stopped_early = True
		break
	sys.stdout.flush()

stats.get_statistics() 