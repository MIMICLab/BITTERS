import argparse, os, datetime, glob, yaml

from bitters.models.vqvae import WaveNet, WaveVAE, WaveGAN
from bitters.loader import ImageDataModule
from bitters.callbacks.image_logger import ReconstructedImageLogger, AdversarialImageLogger

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


    parser = argparse.ArgumentParser(description='AugVAE Training')

    #path configuration
    parser.add_argument('--train_dir', type=str, default='dataset/train/',
                    help='path to train dataset')
    parser.add_argument('--val_dir', type=str, default='dataset/val/',
                    help='path to val dataset')                    
    parser.add_argument('--log_dir', type=str, default='results/',
                    help='path to save logs')
    parser.add_argument('--backup_dir', type=str, default='backups/',
                    help='path to save backups for sudden crash')                    
    parser.add_argument('--ckpt_path', type=str,
                    help='path to previous checkpoint') 
    parser.add_argument('--pretrained_path', type=str,
                    help='path to pretrained codebook') 

    #training configuration
    parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to resume from checkpoint')      
    parser.add_argument('--finetune', action='store_true', default=False,
                    help='finetune pretrained model')                                                       
    parser.add_argument('--backup', action='store_true', default=False,
                    help='save backup and load from backup if restart happens')      
    parser.add_argument('--backup_steps', type =int, default = 1000,
                    help='saves backup every n training steps') 
    parser.add_argument('--log_images', action='store_true', default=False,
                    help='log image outputs. not recommended for tpus')   
    parser.add_argument('--image_log_steps', type=int, default=1000,
                    help='log image outputs for every n step. not recommended for tpus')   
    parser.add_argument('--refresh_rate', type=int, default=1,
                    help='progress bar refresh rate')    
    parser.add_argument('--precision', type=str, default=32,
                    help='precision for training')         

    parser.add_argument('--fake_data', action='store_true', default=False,
                    help='using fake_data for debugging') 
                                                                                        
             
    parser.add_argument('--seed', type=int, default=42,
                    help='random seed')  
    parser.add_argument('--gpus', type=int, default=16,
                    help='number of gpus')     
                          
    parser.add_argument('--num_sanity_val_steps', type=int, default=0,
                    help='num_sanity_val_steps')  
    parser.add_argument('--val_percent_check', type=int, default=100,
                    help='num_val_percent')
    parser.add_argument('--learning_rate', default=1.0e-4, type=float,
                    help='base learning rate')
    parser.add_argument('--lr_decay', action='store_true', default=False,
                    help = 'use learning rate decay')


    parser.add_argument('--batch_size', type=int, default=8,
                    help='training settings')  
    parser.add_argument('--epochs', type=int, default=100,
                    help='training settings')                                    
    parser.add_argument('--num_workers', type=int, default=16,
                    help='training settings')   
    parser.add_argument('--img_size', type=int, default=256,
                    help='training settings')              
    parser.add_argument('--resize_ratio', type=float, default=0.75,
                    help='Random resized crop lower ratio')
                 
    parser.add_argument('--debug', action='store_true', default=False,
                    help='debug run') 
    parser.add_argument('--warmup_percentage', type = float, default=0.01,
                    help = 'Percentage of warmup stage')                       
    #model configuration
    parser.add_argument('--model', type=str, default='pretrain')    
    parser.add_argument('--codebook_dim', type=int, default=64,
                    help='number of embedding dimension for codebook')       
    parser.add_argument('--num_tokens', type=int, default=8192,
                    help='codebook size')        
    parser.add_argument('--resolution', type=int, default=256,
                    help='image resolution')
    parser.add_argument('--in_channels', type=int, default=3,
                    help='input image channel')
    parser.add_argument('--out_channels', type=int, default=3,
                    help='output image channel')    
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='hidden dimension init size')  
    parser.add_argument('--num_res_blocks', type=int, default=16,
                    help='number of resnet blocks')                     
                              
    #loss configuration
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--p_loss_weight', type = float, default=0.1,
                    help = 'Perceptual loss weight')                    
    parser.add_argument('--disc_weight', type = float, default=0.1,
                    help = 'Adversarial loss weight')    
    #misc configuration
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    
    #random seed fix
    seed_everything(args.seed)   

    tpus = None
    gpus = args.gpus      
    args.world_size = args.gpus    
    
    datamodule = ImageDataModule(args.train_dir, args.val_dir, 
                            args.batch_size, args.num_workers, 
                            args.img_size, (args.model=='pretrain'), args.resize_ratio)
    datamodule.setup()

    args.steps_per_epoch = len(datamodule.train_dataloader()) // args.world_size + 1
    # model           
    if args.model == 'gan':
        model = WaveGAN.load_from_checkpoint(args.pretrained_path, args=args, 
                                            batch_size=args.batch_size, 
                                            learning_rate=args.learning_rate)                     
        model.setup_gan(args)  
    elif args.model == 'vae':
        model = WaveVAE.load_from_checkpoint(args.pretrained_path, args=args, 
                                            batch_size=args.batch_size, 
                                            learning_rate=args.learning_rate)                     
        model.setup_vae(args)                  
    else:              
        model = WaveNet(args, args.batch_size, args.learning_rate)                         

    default_root_dir = args.log_dir

    if args.debug:
        limit_train_batches = 100
        limit_test_batches = 100
        args.backup_steps = 10
        args.image_log_steps = 10
    else:
        limit_train_batches = 1.0
        limit_test_batches = 1.0   

    if args.resume:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = None
    if args.model == 'pretrain':
        target_loss = 'total_loss'
    else:
        target_loss = 'rec_loss'

    if args.val_percent_check ==0:
        checkpoint_callback = ModelCheckpoint(monitor=f"train/{target_loss}")        
    else:
        checkpoint_callback = ModelCheckpoint(monitor=f"val/{target_loss}")
    
    if args.model == 'gan':
        model_name = 'wavegan'
    elif args.model == 'vae':
        model_name = 'wavevae'
    else:
        model_name = 'wavenet'

    if args.backup:
        args.backup_dir = os.path.join(args.backup_dir, f'vae/{model_name}')
        backup_callback = ModelCheckpoint(
                                    dirpath=args.backup_dir,
                                    every_n_train_steps = args.backup_steps,
                                    filename='{epoch}_{step}'
                                    )
        
        if len(glob.glob(os.path.join(args.backup_dir,'*.ckpt'))) != 0 :
            ckpt_path = sorted(glob.glob(os.path.join(args.backup_dir,'*.ckpt')))[-1]
            if args.resume:
                print("Setting default ckpt to {}. If this is unexpected behavior, remove {}".format(ckpt_path, ckpt_path))

    logger = pl.loggers.tensorboard.TensorBoardLogger(args.log_dir, name=f'{model_name}')     

    trainer = Trainer(devices=gpus,  
                      strategy = DDPStrategy(find_unused_parameters=(args.model=='gan')),      
                      accelerator="gpu", 
                      max_epochs=args.epochs, 
                      precision=args.precision, benchmark=True,                        
                      num_sanity_val_steps=args.num_sanity_val_steps,
                      limit_val_batches = args.val_percent_check,                     
                      limit_train_batches=limit_train_batches,
                      limit_test_batches=limit_test_batches,                          
                      callbacks=[checkpoint_callback],   
                      logger = logger)

    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))
    if args.backup:
        trainer.callbacks.append(backup_callback)  
    if args.resume:    
        trainer.callbacks.append(ModelCheckpoint())                                     
    if args.log_images:
        if args.model == 'pretrain':
            trainer.callbacks.append(ReconstructedImageLogger(every_n_steps=args.image_log_steps, nrow=args.batch_size))
        else:            
            trainer.callbacks.append(AdversarialImageLogger(every_n_steps=args.image_log_steps, nrow=args.batch_size))            
  

    print("Setting batch size: {} learning rate: {:.2e}".format(model.hparams.batch_size, model.hparams.learning_rate))
        
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


