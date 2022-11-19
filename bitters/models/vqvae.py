import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from bitters.modules.vqvae.vae import WaveEncoder, WaveDecoder
from bitters.modules.vqvae.quantize import VectorQuantizer
from bitters.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from torch.optim.lr_scheduler import OneCycleLR
from einops import rearrange
from bitters.utils.util import normalize
from torchvision import transforms as T
from pytorch_lightning.utilities.distributed import rank_zero_only

class WaveNet(pl.LightningModule):
    def __init__(self, args, batch_size, learning_rate):
        super().__init__()   
        self.image_size = args.resolution
        self.num_tokens = args.num_tokens
        self.patch_size = int(np.sqrt(args.codebook_dim))

        self.quantizer = VectorQuantizer(num_tokens=args.num_tokens,
                                       codebook_dim=args.codebook_dim)

        self.encoder = WaveEncoder(args.hidden_dim, args.in_channels, args.codebook_dim, 
                                  args.num_res_blocks)

        self.decoder = WaveDecoder(args.hidden_dim, args.codebook_dim, args.out_channels, 
                                  args.num_res_blocks)

        self.setup_loss(args)

        self.save_hyperparameters("args", "batch_size", "learning_rate")
        self.args = args  
        self.image_seq_len = (args.resolution // 8) ** 2 

    def setup_loss(self, args):
        if args.loss_type == 'smooth_l1':
            self.loss = nn.SmoothL1Loss()

        elif args.loss_type == 'l1':
            self.loss = nn.L1Loss()
        
        elif args.loss_type == 'mse':
            self.loss = nn.MSELoss()            
        else:
            print(f"Loss type {args.loss_type} is not currently supported. Using default MSELoss.")
            self.loss = nn.MSELoss()  

    def setup_eval(self):
        self.freeze()
        del self.loss
        
    def encode(self, input, x_high=None, x_mid=None, x_low=None):
        if self.encoder.auxiliary:
            enc_1, enc_2, enc_3, enc_4 = self.encoder(input, x_high, x_mid, x_low)
            qloss_1, z_q_1, indices_1 = self.quantizer(enc_1)
            qloss_2, z_q_2, indices_2 = self.quantizer(enc_2)
            qloss_3, z_q_3, indices_3 = self.quantizer(enc_3)  
            qloss_4, z_q_4, indices_4 = self.quantizer(enc_4)                      
            return qloss_1 + qloss_2 + qloss_3 + qloss_4, [z_q_1, z_q_2, z_q_3, z_q_4], [indices_1, indices_2, indices_3, indices_4]
        else:
            enc = self.encoder(input)
            qloss, z_q, indices = self.quantizer(enc)
            return qloss, z_q, indices

    @torch.no_grad()
    def get_codebook_indices(self, img):
        assert self.encoder.auxiliary == False
        b = img.shape[0]
        _, _, indices = self.encode(img)
        n = torch.div(indices.shape[0], b, rounding_mode='trunc')
        indices = indices.view(b,n)     
        return indices

    def decode(self, input, z_low=None, z_mid=None, z_high=None, feed_seq=False):
        if self.encoder.auxiliary:
            z_q = input
            out_1, out_2, out_3, out_4 = self.decoder(z_q, z_low, z_mid, z_high)
            return out_1, out_2, out_3, out_4
        else:
            if feed_seq:
                z_q = self.quantizer.embedding(input) 
                b, n, c = z_q.shape
                h = w = int(math.sqrt(n))            
                z_q = rearrange(z_q, 'b (h w) c -> b c h w', h = h, w = w).contiguous()                
            else:
                z_q = input
            out = self.decoder(z_q)
            return out

    def forward(self, x, x_high, x_mid, x_low):
        qloss, z_q, _ = self.encode(x, x_high, x_mid, x_low)
        out_1, out_2, out_3, out_4 = self.decode(z_q[-1], z_q[-2], z_q[-3], z_q[-4])
        return qloss, out_1, out_2, out_3, out_4

    def configure_optimizers(self):
        lr = self.hparams.learning_rate   
        params = []
        params += list(self.quantizer.parameters())
        params += list(self.encoder.parameters()) 
        params += list(self.decoder.parameters())   
        opt = torch.optim.AdamW(set(params), lr=lr, betas=(0.9, 0.999),weight_decay=1e-5, eps=1e-4)
        if self.args.lr_decay:
            scheduler = OneCycleLR(opt,
                                    max_lr=self.hparams.learning_rate,
                                    epochs=self.args.epochs,
                                    steps_per_epoch=self.args.steps_per_epoch,
                                    pct_start=self.args.warmup_percentage)
            sched = {'scheduler': scheduler, 'interval': 'step'}
            return [opt], [sched]
        else:
            return [opt], []

    def training_step(self, batch, batch_idx):     
        x_low, x_mid, x_high, x = batch
        qloss, xrec_low, xrec_mid, xrec_high, xrec = self(x, x_high, x_mid, x_low)   
        lowloss = self.loss(xrec_low, x_low)                       
        midloss = self.loss(xrec_mid, x_mid) 
        highloss = self.loss(xrec_high, x_high)    
        recloss = self.loss(xrec, x)         
        loss = qloss + lowloss + midloss + highloss + recloss     
        self.log("train/rec_loss", recloss, prog_bar=False, logger=True)                        
        self.log("train/rec_high_loss", highloss, prog_bar=False, logger=True)
        self.log("train/rec_mid_loss", midloss, prog_bar=False, logger=True)            
        self.log("train/rec_low_loss", lowloss, prog_bar=False, logger=True)                
        self.log("train/quantization_loss", qloss, prog_bar=False, logger=True) 

        self.log("train/total_loss", loss, prog_bar=False, logger=True)                    
        if self.args.log_images:
            return {'loss':loss,                     
                    'xrec':xrec.detach(),                                      
                    'xrec_high':xrec_high.detach(), 
                    'xrec_mid':xrec_mid.detach(),
                    'xrec_low':xrec_low.detach()}            
        else:
            return loss

    def validation_step(self, batch, batch_idx):       
        x_low, x_mid, x_high, x = batch
        qloss, xrec_low, xrec_mid, xrec_high, xrec = self(x, x_high, x_mid, x_low)  
        lowloss = self.loss(xrec_low, x_low)                       
        midloss = self.loss(xrec_mid, x_mid) 
        highloss = self.loss(xrec_high, x_high)    
        recloss = self.loss(xrec, x)         
        loss = qloss + lowloss + midloss + highloss + recloss     
        self.log("val/rec_loss", recloss, prog_bar=False, logger=True)                        
        self.log("val/rec_high_loss", highloss, prog_bar=False, logger=True)
        self.log("val/rec_mid_loss", midloss, prog_bar=False, logger=True)            
        self.log("val/rec_low_loss", lowloss, prog_bar=False, logger=True)                
        self.log("val/quantization_loss", qloss, prog_bar=False, logger=True) 

        self.log("val/total_loss", loss, prog_bar=False, logger=True)   

        if self.args.log_images:
            return {'loss':loss,                      
                    'xrec':xrec.detach(),                                           
                    'xrec_high':xrec_high.detach(), 
                    'xrec_mid':xrec_mid.detach(),
                    'xrec_low':xrec_low.detach()}            
        else:
            return loss
   
class WaveVAE(WaveNet):
    def __init__(self,args, batch_size, learning_rate):
        super().__init__(args, batch_size, learning_rate)   

    def setup_vae(self, args):  
        self.quantizer.freeze()  
        self.encoder.freeze()
        self.decoder.freeze() 
        self.encoder.setup_finetune()
        self.decoder.setup_finetune()

    def forward(self, x):
        qloss, z_q, _ = self.encode(x)
        out = self.decode(z_q)
        return qloss, out

    def training_step(self, batch, batch_idx):     
        x = batch
        qloss, xrec = self(x)         
        recloss = self.loss(x, xrec) 
        loss = qloss + recloss
        self.log("train/rec_loss", recloss, prog_bar=True, logger=True)              
        self.log("train/quantization_loss", qloss, prog_bar=False, logger=True)               
        self.log("train/total_loss", loss, prog_bar=False, logger=True) 

        if self.args.log_images:
            return {'loss':loss, 'x':x.detach(), 'xrec':xrec.detach()}

        return loss

    def validation_step(self, batch, batch_idx):     
        x = batch
        qloss, xrec = self(x)         
        recloss = self.loss(x, xrec) 
        loss = qloss + recloss
        self.log("val/rec_loss", recloss, prog_bar=True, logger=True)              
        self.log("val/quantization_loss", qloss, prog_bar=False, logger=True)               
        self.log("val/total_loss", loss, prog_bar=False, logger=True) 

        if self.args.log_images:
            return {'loss':loss, 'x':x.detach(), 'xrec':xrec.detach()}

        return loss


class WaveGAN(WaveVAE):
    def __init__(self,args, batch_size, learning_rate):
        super().__init__(args, batch_size, learning_rate)   
        self.setup_vae(args)
        

    def setup_gan(self, args): 
        self.quantizer.unfreeze()
        self.encoder.unfreeze()
        self.decoder.unfreeze()
        del self.loss
        self.loss = VQLPIPSWithDiscriminator(disc_weight = args.disc_weight, 
                                            perceptual_weight=args.p_loss_weight)  
    def setup_eval(self):
        self.quantizer.freeze()
        self.encoder.freeze()
        self.decoder.freeze()
        del self.loss

    def forward(self, x):
        qloss, z_q, _ = self.encode(x)
        out = self.decode(z_q)
        return qloss, out

    def configure_optimizers(self):
        lr = self.hparams.learning_rate  
        params = []
        params += list(self.quantizer.parameters())
        params += list(self.encoder.parameters()) 
        params += list(self.decoder.parameters())   

        opt_ae = torch.optim.Adam(set(params),
                                lr=lr, betas=(0.9, 0.999))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                lr=lr, betas=(0.9, 0.999))

        scheduler_ae = OneCycleLR(opt_ae,
                                max_lr=self.hparams.learning_rate,
                                epochs=self.args.epochs,
                                steps_per_epoch=self.args.steps_per_epoch,
                                pct_start=self.args.warmup_percentage)
        sched_ae = {'scheduler': scheduler_ae, 'interval': 'step'}
        scheduler_disc = OneCycleLR(opt_disc,
                                max_lr=self.hparams.learning_rate,
                                epochs=self.args.epochs,
                                steps_per_epoch=self.args.steps_per_epoch,
                                pct_start=self.args.warmup_percentage)
        sched_disc = {'scheduler': scheduler_disc, 'interval': 'step'}        
        return [opt_ae, opt_disc], [sched_ae, sched_disc]                   

    def training_step(self, batch, batch_idx, optimizer_idx=0):     
        x = batch
        qloss, xrec = self(x)
        if optimizer_idx == 0:
            recloss = F.l1_loss(xrec, x)            
            loss = qloss + self.loss(x, xrec, optimizer_idx)
            self.log("train/rec_loss", recloss, prog_bar=True, logger=True)              
            self.log("train/g_loss", loss, prog_bar=True, logger=True)   
            self.log("train/quantization_loss", qloss, prog_bar=False, logger=True)               
            self.log("train/total_loss", loss, prog_bar=False, logger=True) 

        elif optimizer_idx == 1:  
            loss = self.loss(x, xrec, optimizer_idx)                                            
            self.log("train/d_loss", loss, prog_bar=True,logger=True)

        if self.args.log_images:
            return {'loss':loss, 'x':x.detach(), 'xrec':xrec.detach()}

        return loss

    def validation_step(self, batch, batch_idx):     
        x = batch

        qloss, xrec = self(x)
        aeloss = qloss + self.loss(x, xrec, 0)
        discloss = self.loss(x, xrec, 1) 
        recloss = F.l1_loss(xrec, x) 
        self.log("val/rec_loss", recloss, prog_bar=True, logger=True)    
        self.log("val/quantization_loss", qloss, prog_bar=False, logger=True)                                                                   
        self.log("val/g_loss", aeloss, prog_bar=True, logger=True)
        self.log("val/d_loss", discloss, prog_bar=True,logger=True)                               
                         
        loss = aeloss

        self.log("val/total_loss", loss, prog_bar=False, logger=True) 

        if self.args.log_images:
            return {'loss':loss, 'x':x.detach(), 'xrec':xrec.detach()}

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx): 
        if self.decoder == None:
            text, image = batch
            image_indices = self.get_codebook_indices(image)
            return {'encoded_texts':text, 'encoded_images':image_indices} 
        else:
            x = batch
            x_ref = normalize(x)
            _, x_rec = self(x)         
            x_rec = normalize(x_rec)

            input_images = x_ref
            generated_images = x_rec     

            return {'input_images':input_images, 'generated_images':generated_images} 
 
    @torch.no_grad()
    def gather_and_save(self, outputs, batch_idx):
        # this out is now the full size of the batch        
        outputs = self.all_gather(outputs)
        originals = []          
        generated = []

        x_i_refs = outputs["input_images"]
        for images in x_i_refs:
            for img in images:
                originals.append(img)       
        x_i_gens = outputs["generated_images"]
        for images in x_i_gens:
            for img in images:
                generated.append(img)
      
        self.save_results(self.args.log_dir, batch_idx, originals, generated)

    @torch.no_grad()
    def gather_and_save_idxs(self, outputs, batch_idx):
        # this out is now the full size of the batch        
        outputs = self.all_gather(outputs)
        text_indices = []          
        image_indices = []

        x_i_texts = outputs["encoded_texts"]
        for texts in x_i_texts:
            for text in texts:
                text_indices.append(text)       
        x_i_images = outputs["encoded_images"]
        for images in x_i_images:
            for img in images:
                image_indices.append(img)
        result_dir = os.path.join(self.args.log_dir,'encoded_texts/')
        os.makedirs(result_dir,exist_ok=True)
        for idx, text in enumerate(text_indices):
            text = text.masked_select(text != 0).tolist()
            text = ' '.join([ str(int(x)) for x in text ])            
            fp =  open(os.path.join(result_dir,'{}_{}.txt'.format(batch_idx,idx)), 'w')     
            print(text,file=fp)
            fp.close()
  
        result_dir = os.path.join(self.args.log_dir,'encoded_images/')
        os.makedirs(result_dir,exist_ok=True)
        for idx, text in enumerate(image_indices):
            text = text.tolist()
            text = ' '.join([ str(int(x)) for x in text ])  
            fp =  open(os.path.join(result_dir,'{}_{}.txt'.format(batch_idx,idx)), 'w')     
            print(text,file=fp)
            fp.close()

    @rank_zero_only
    def save_results(self, log_dir, batch_idx, original_images, generated):
        result_dir = os.path.join(log_dir,'original_images/')
        os.makedirs(result_dir,exist_ok=True)
        for idx, image in enumerate(original_images):
            image = T.ToPILImage()(image)
            image.save(os.path.join(result_dir,'{}_{}.png'.format(batch_idx, idx)))
  
        result_dir = os.path.join(log_dir,'generated_images/')
        os.makedirs(result_dir,exist_ok=True)
        for idx, image in enumerate(generated):
            image = T.ToPILImage()(image)
            image.save(os.path.join(result_dir,'{}_{}.png'.format(batch_idx, idx)))

 