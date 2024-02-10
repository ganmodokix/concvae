from typing import Dict

from vaetc.utils import debug_print

from .abstract import VisionLanguageEncoder
# from .vse import VSE
from .clip import CLIP

pretrained_vlmodel_cache: Dict[str, VisionLanguageEncoder] = {}

# get a pre-trained model as a singleton object
def get_pretrained_vlmodel(name: str):

    global pretrained_vlmodel_cache

    if name not in pretrained_vlmodel_cache:
        
        debug_print(f"Loading model '{name}' ...")

        if name == "vse0":
            # vlmodel = VSE(run_name="coco_vse0_vggfull_restval_finetune")
            # vlmodel = VSE(run_name="coco_vse0_resnet_restval_finetune")
            raise NotImplementedError()
        elif name == "vse++":
            # vlmodel = VSE(run_name="coco_vse++_vggfull_restval_finetune")
            # vlmodel = VSE(run_name="coco_vse++_resnet_restval_finetune")
            # raise ValueError()
            raise NotImplementedError()
        elif name == "clip":
            vlmodel = CLIP(name="ViT-B/32")
        elif name == "clip_scratch":
            vlmodel = CLIP(name="ViT-B/32", pretrained=False)
        elif name == "clip_vitb16":
            vlmodel = CLIP(name="ViT-B/16")
        elif name == "clip_rn50":
            vlmodel = CLIP(name="RN50")
        elif name == "clip_rn101":
            vlmodel = CLIP(name="RN101")
        else:
            raise NotImplementedError(f"Pre-trained model '{name}' is not implemented")
        
        pretrained_vlmodel_cache[name] = vlmodel.cuda()
        
    pretrained_vlmodel_cache[name].eval()

    return pretrained_vlmodel_cache[name]