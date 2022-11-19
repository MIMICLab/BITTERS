from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning import Callback
from bitters.utils.util import normalize
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF


class ReconstructedImageLogger(Callback):
    def __init__(
        self,
        every_n_steps: int = 1000,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``True``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        super().__init__()
        self.every_n_steps = every_n_steps
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0          
    ) -> None:
        """Called when the validation batch ends."""
        if batch_idx % self.every_n_steps == 0:
            x_low, x_mid, x_high, x = batch

            x = normalize(x)
            x_high = normalize(x_high)
            x_mid = normalize(x_mid)                
            x_low = normalize(x_low)
            xrec = normalize(outputs['xrec'])            
            xrec_low = normalize(outputs['xrec_low'])            
            xrec_high = normalize(outputs['xrec_high'])
            xrec_mid = normalize(outputs['xrec_mid'])    

            x_grid = torchvision.utils.make_grid(
                    tensor=x,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                ) 
            x_high_grid = torchvision.utils.make_grid(
                    tensor=x_high,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )    
            x_mid_grid = torchvision.utils.make_grid(
                    tensor=x_mid,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )                         
            x_low_grid = torchvision.utils.make_grid(
                    tensor=x_low,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )   
            xrec_grid = torchvision.utils.make_grid(
                    tensor=xrec,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )                          
            xrec_high_grid = torchvision.utils.make_grid(
                    tensor=xrec_high,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )    
            xrec_mid_grid = torchvision.utils.make_grid(
                    tensor=xrec_mid,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )                         
            xrec_low_grid = torchvision.utils.make_grid(
                    tensor=xrec_low,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )  

            x_title = "train/3_x_raw"
            trainer.logger.experiment.add_image(x_title, x_grid, global_step=trainer.global_step)
            x_high_title = "train/2_x_high"
            trainer.logger.experiment.add_image(x_high_title, x_high_grid, global_step=trainer.global_step)
            x_mid_title = "train/1_x_mid"
            trainer.logger.experiment.add_image(x_mid_title, x_mid_grid, global_step=trainer.global_step)
            x_low_title = "train/0_x_low"
            trainer.logger.experiment.add_image(x_low_title, x_low_grid, global_step=trainer.global_step)  
            
            xrec_title = "train/3_xrec_raw"
            trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=trainer.global_step)                    
            xrec_high_title = "train/2_xrec_high"
            trainer.logger.experiment.add_image(xrec_high_title, xrec_high_grid, global_step=trainer.global_step)
            xrec_mid_title = "train/1_xrec_mid"
            trainer.logger.experiment.add_image(xrec_mid_title, xrec_mid_grid, global_step=trainer.global_step)
            xrec_low_title = "train/0_xrec_low"
            trainer.logger.experiment.add_image(xrec_low_title, xrec_low_grid, global_step=trainer.global_step)
    
    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0     
    ) -> None:
        """Called when the validation batch ends."""
        if batch_idx % self.every_n_steps == 0:
            x_low, x_mid, x_high, x = batch

            x = normalize(x)
            x_high = normalize(x_high)
            x_mid = normalize(x_mid)                
            x_low = normalize(x_low)
            xrec = normalize(outputs['xrec'])            
            xrec_low = normalize(outputs['xrec_low'])            
            xrec_high = normalize(outputs['xrec_high'])
            xrec_mid = normalize(outputs['xrec_mid'])    

            x_grid = torchvision.utils.make_grid(
                    tensor=x,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                ) 
            x_high_grid = torchvision.utils.make_grid(
                    tensor=x_high,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )    
            x_mid_grid = torchvision.utils.make_grid(
                    tensor=x_mid,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )                         
            x_low_grid = torchvision.utils.make_grid(
                    tensor=x_low,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )   
            xrec_grid = torchvision.utils.make_grid(
                    tensor=xrec,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )                          
            xrec_high_grid = torchvision.utils.make_grid(
                    tensor=xrec_high,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )    
            xrec_mid_grid = torchvision.utils.make_grid(
                    tensor=xrec_mid,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )                         
            xrec_low_grid = torchvision.utils.make_grid(
                    tensor=xrec_low,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )  

            x_title = "val/3_x_raw"
            trainer.logger.experiment.add_image(x_title, x_grid, global_step=trainer.global_step)
            x_high_title = "val/2_x_high"
            trainer.logger.experiment.add_image(x_high_title, x_high_grid, global_step=trainer.global_step)
            x_mid_title = "val/1_x_mid"
            trainer.logger.experiment.add_image(x_mid_title, x_mid_grid, global_step=trainer.global_step)
            x_low_title = "val/0_x_low"
            trainer.logger.experiment.add_image(x_low_title, x_low_grid, global_step=trainer.global_step)  
            
            xrec_title = "val/3_xrec_raw"
            trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=trainer.global_step)                    
            xrec_high_title = "val/2_xrec_high"
            trainer.logger.experiment.add_image(xrec_high_title, xrec_high_grid, global_step=trainer.global_step)
            xrec_mid_title = "val/1_xrec_mid"
            trainer.logger.experiment.add_image(xrec_mid_title, xrec_mid_grid, global_step=trainer.global_step)
            xrec_low_title = "val/0_xrec_low"
            trainer.logger.experiment.add_image(xrec_low_title, xrec_low_grid, global_step=trainer.global_step)
     
class AdversarialImageLogger(Callback):
    def __init__(
        self,
        every_n_steps: int = 1000,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        use_wandb: bool = False
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``True``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        super().__init__()
        self.every_n_steps = every_n_steps
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.use_wandb = use_wandb

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0             
    ) -> None:
        """Called when the train batch ends."""

        if batch_idx % self.every_n_steps == 0:
            try:
                x = outputs[0]['x']
                xrec = outputs[0]['xrec']
            except:
                x = outputs['x']
                xrec = outputs['xrec']                

            x = normalize(x)
            xrec = normalize(xrec)

            x_grid = torchvision.utils.make_grid(
                    tensor=x,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )           
            xrec_grid = torchvision.utils.make_grid(
                    tensor=xrec,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )  

            x_title = "train/input"
            trainer.logger.experiment.add_image(x_title, x_grid, global_step=trainer.global_step)
            xrec_title = "train/reconstruction"
            trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=trainer.global_step)

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0          
    ) -> None:
        """Called when the validation batch ends."""
        if batch_idx % self.every_n_steps == 0:   
            try:
                x = outputs[0]['x']
                xrec = outputs[0]['xrec']
            except:
                x = outputs['x']
                xrec = outputs['xrec']   

            x = normalize(x)
            xrec = normalize(xrec)

            x_grid = torchvision.utils.make_grid(
                    tensor=x,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )           
            xrec_grid = torchvision.utils.make_grid(
                    tensor=xrec,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )  
 
            x_title = "val/input"
            trainer.logger.experiment.add_image(x_title, x_grid, global_step=trainer.global_step)
            xrec_title = "val/reconstruction"
            trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=trainer.global_step)

    def on_test_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0           
    ) -> None:
        """Called when the text batch ends.""" 
        if pl_module.decoder == None:
            pl_module.gather_and_save_idxs(outputs, batch_idx)
            return outputs      
                               
        x = outputs['input_images']
        xrec = outputs['generated_images']

        x = normalize(x)
        xrec = normalize(xrec)

        x_grid = torchvision.utils.make_grid(
                tensor=x,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )           
        xrec_grid = torchvision.utils.make_grid(
                tensor=xrec,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )  

        x_title = "test/input"
        trainer.logger.experiment.add_image(x_title, x_grid, global_step=batch_idx)
        xrec_title = "test/reconstruction"
        trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=batch_idx)
        
        pl_module.gather_and_save(outputs, batch_idx)
        return outputs
