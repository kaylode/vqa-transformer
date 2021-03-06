from utils.getter import *
import argparse

parser = argparse.ArgumentParser('Training Object Detection')
parser.add_argument('--print_per_iter', type=int, default=300, help='Number of iteration to print')
parser.add_argument('--val_interval', type=int, default=2, help='Number of epoches between valing phases')
parser.add_argument('--save_interval', type=int, default=1000, help='Number of steps between saving')
parser.add_argument('--resume', type=str, default=None,
                    help='whether to load weights from a checkpoint, set None to initialize')
parser.add_argument('--saved_path', type=str, default='./weights')
parser.add_argument('--no_visualization', action='store_false', help='whether to visualize box to ./sample when validating (for debug), default=on')
parser.add_argument('--bottom-up', action='store_true', help='use bottom-up attention, must provided npy_path in config')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
seed_everything()

def train(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    devices_info = get_devices_info(config.gpu_devices)
    
    trainset, valset, trainloader, valloader = get_dataset_and_dataloader(config, args.bottom_up)

    if args.bottom_up:
        net = get_transformer_bottomup_model(
            bottom_up_dim=trainset.get_feature_dim(),
            trg_vocab=trainset.tokenizer.vocab_size,
            num_classes=trainset.num_classes)
    else:
        net = get_transformer_model(
            trg_vocab=trainset.tokenizer.vocab_size, 
            num_classes=trainset.num_classes)

    optimizer, optimizer_params = get_lr_policy(config.lr_policy)

    criterion = LabelSmoothing(smoothing=0.3)

    model = Captioning(
            model = net,
            criterion=criterion,
            metrics=AccuracyMetric(valloader, max_samples = None),
            scaler=NativeScaler(),
            optimizer= optimizer,
            optim_params = optimizer_params,     
            device = device)

    if args.resume is not None:                
        load_checkpoint(model, args.resume)
        start_epoch, start_iter, best_value = get_epoch_iters(args.resume)
    else:
        print('Not resume. Load pretrained weights...')
        # args.resume = os.path.join(CACHE_DIR, f'{config.model_name}.pth')
        # download_pretrained_weights(f'{config.model_name}', args.resume)
        # load_checkpoint(model, args.resume)
        start_epoch, start_iter, best_value = 0, 0, 0.0
        
    scheduler, step_per_epoch = get_lr_scheduler(
        model.optimizer, train_len=len(trainloader),
        lr_config=config.lr_scheduler,
        num_epochs=config.num_epochs)

    if args.resume is not None:                 
        old_log = find_old_log(args.resume)
    else:
        old_log = None

    args.saved_path = os.path.join(
        args.saved_path, 
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    trainer = Trainer(config,
                     model,
                     trainloader, 
                     valloader,
                     checkpoint = Checkpoint(save_per_iter=args.save_interval, path = args.saved_path),
                     best_value=best_value,
                     logger = Logger(log_dir=args.saved_path, resume=old_log),
                     scheduler = scheduler,
                     evaluate_per_epoch = args.val_interval,
                     visualize_when_val = args.no_visualization,
                     step_per_epoch = step_per_epoch)
    print()
    print("##########   DATASET INFO   ##########")
    print("Trainset: ")
    print(trainset)
    print("Valset: ")
    print(valset)
    print()
    print(trainer)
    print()
    print(config)
    print(f'Training with {num_gpus} gpu(s): ')
    print(devices_info)
    print(f"Start training at [{start_epoch}|{start_iter}]")
    print(f"Current best BLEU: {best_value}")

    trainer.fit(start_epoch = start_epoch, start_iter = start_iter, num_epochs=config.num_epochs, print_per_iter=args.print_per_iter)

    

if __name__ == '__main__':
    
    args = parser.parse_args()
    config = Config(os.path.join('configs','config.yaml'))

    train(args, config)