from data import get_dataset, get_dataloaders
from model import EfficientDetModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger


if __name__ == '__main__':

	parser.add_argument('-r', '--run_name', type=str, help="name for wandb run", required=True)
	parser.add_argument('--dir_input', type=str, help="directory for csv", required=True)
	parser.add_argument('--dir_train', type=str, help="directory for images", required=True)
	parser.add_argument('--debug', action='store_true', default = False,  help='use in debug mode')

	parser.add_argument('--num_classes', type=int, required = True, help='number of classes')
	parser.add_argument('--batch_size', type=int, default=4, help='batch_size.')
	parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
	parser.add_argument('--patience', type=int, default=5, help='rlp patience')
	parser.add_argument('--factor', type=float, default=0.1, help='rlp factor')
	parser.add_argument('--max_epochs', type=int, default=50, help='epochs to train')

	args = parser.parse_args()

	seed_everything(42)

	traindf, valdf = get_dataset(args)
	trainloader, valloader = get_dataloaders(args, traindf, valdf)

	model = EfficientDetModel(args)

	wandblogger = WandbLogger(project = 'object_detection', config = args, name = args.run_name) 
	checkpoint_callback = ModelCheckpoint(
	    filepath=f'./{args.run_name}',
	    verbose=True,
	    monitor='val_loss',
	    mode='min',
	    save_weights_only = True
	)


	trainer = Trainer(logger = wandblogger, max_epochs = args.max_epochs, gpus = 1) 

	trainer.fit(model, trainloader, valloader)