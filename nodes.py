# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import warnings
warnings.filterwarnings('ignore', module="torchvision")

from PIL import Image
import os
from datetime import datetime
import folder_paths
import comfy
from comfy import samplers
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

import numpy as np

#comfy essentials
def p(image):
    return image.permute([0,3,1,2])
def pb(image):
    return image.permute([0,2,3,1])

# Tensor to PIL (WAS Node)
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor (WAS Node)
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

FLOAT = ("FLOAT", {"default": 1,
                   "min": -sys.float_info.max,
                   "max": sys.float_info.max,
                   "step": 0.01})

BOOLEAN = ("BOOLEAN", {"default": True})
BOOLEAN_FALSE = ("BOOLEAN", {"default": False})

INT = ("INT", {"default": 1,
               "min": -sys.maxsize,
               "max": sys.maxsize,
               "step": 1})

STRING = ("STRING", {"default": ""})


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


ANY = AnyType("*")

SCHEDULERS_COMFY = comfy.samplers.KSampler.SCHEDULERS
SCHEDULERS_EFFICIENT = comfy.samplers.KSampler.SCHEDULERS + ['AYS SD1', 'AYS SDXL', 'AYS SVD']
SCHEDULERS_IMPACT = comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]']
#---------------------------------------------------------------------------------------------------------------------#
#imported from crystools
class CBoolean:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": BOOLEAN,
            }
        }

    CATEGORY = "Rvaged/Primitives"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"

    def execute(self, boolean=True):
        return (boolean,)

#---------------------------------------------------------------------------------------------------------------------#    
class CFloat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": FLOAT,
            }
        }

    CATEGORY = "Rvaged/Primitives"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "execute"

    def execute(self, float=True):
        return (float,)
    
#---------------------------------------------------------------------------------------------------------------------#    
class CInteger:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": INT,
            }
        }

    CATEGORY = "Rvaged/Primitives"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "execute"

    def execute(self, int=True):
        return (int,)

#---------------------------------------------------------------------------------------------------------------------#    
class CText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": STRING,
            }
        }

    CATEGORY = "Rvaged/Primitives"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"

    def execute(self, string=""):
        return (string,)

#---------------------------------------------------------------------------------------------------------------------#
def format_datetime(datetime_format):
    today = datetime.now()
    return f"{today.strftime(datetime_format)}"

#---------------------------------------------------------------------------------------------------------------------#
#imported from path-helper
def format_date_time(string, position, datetime_format):
    today = datetime.now()
    if position == "prefix":
        return f"{today.strftime(datetime_format)}_{string}"
    if position == "postfix":
        return f"{string}_{today.strftime(datetime_format)}"

#---------------------------------------------------------------------------------------------------------------------#
def format_variables(string, input_variables):
    if input_variables:
        variables = str(input_variables).split(",")
        return string.format(*variables)
    else:
        return string

#---------------------------------------------------------------------------------------------------------------------#
#altered from CreateRootFolder from path-helper
class CreateProjectFolder:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "date_time_format": ("STRING", {"multiline": False, "default": "%Y-%m-%d"}),
                "add_date_time": (["disable", "prefix", "postfix"],),
                "project_root_name": ("STRING", {"multiline": False, "default": "MyProject"}),
                "create_batch_folder": (["enable", "disable"],),
                "batch_folder_name": ("STRING", {"multiline": False, "default": "batch_{}"}),                
                "output_path_generation": (["relative", "absolute"],)
            },
            "optional": {
                "input_variables": (ANY,)
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "create_project_folder"
    CATEGORY = "Rvaged/Folder"

    def create_project_folder(self, project_root_name, add_date_time, date_time_format, output_path_generation, create_batch_folder, batch_folder_name, input_variables=None):
        mDate = format_datetime(date_time_format)
        new_path = project_root_name

        if add_date_time == "prefix":
            new_path = os.path.join(mDate, project_root_name)
        elif add_date_time == "postfix":
            new_path = os.path.join(project_root_name, mDate)

        if create_batch_folder == "enable":
           folder_name_parsed = format_variables(batch_folder_name, input_variables)
           new_path = os.path.join(new_path, folder_name_parsed)

        if output_path_generation == "relative":
            return ("./" + new_path,)
        elif output_path_generation == "absolute":
            return (os.path.join(self.output_dir, new_path),)
        
#---------------------------------------------------------------------------------------------------------------------#
class Add_Folder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"forceInput": True}),
                #"path": ("PATH",),
                "folder_name": ("STRING", {"multiline": False, "default": "SubFolder"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)    
    FUNCTION = "add_folder"
    CATEGORY = "Rvaged/Folder"

    def add_folder(self, path, folder_name):
        new_path = os.path.join(path, folder_name)
        return (new_path,)

#---------------------------------------------------------------------------------------------------------------------#
class Add_FileNamePrefix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"forceInput": True}),
                #"path": ("PATH",),
                "file_name_prefix": ("STRING", {"multiline": False, "default": "image"}),
                "add_date_time": (["disable", "prefix", "postfix"],),
                "date_time_format": ("STRING", {"multiline": False, "default": "%Y-%m-%d_%H:%M:%S"}),
            },
            "optional": {
                "input_variables": (ANY,)
            }
        }

    CATEGORY = "Rvaged/Folder"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "add_filename_prefix"

    def add_filename_prefix(self, path, file_name_prefix, add_date_time, date_time_format, input_variables=None):
        filename_name_parsed = format_variables(file_name_prefix, input_variables)
        if add_date_time == "disable":
            new_path = os.path.join(path, filename_name_parsed)
        else:
            new_path = os.path.join(path, format_date_time(filename_name_parsed, add_date_time, date_time_format))
        return (new_path,)

#---------------------------------------------------------------------------------------------------------------------#
class Join_Vars:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "var_1": (ANY,),
            },
            "optional": {
                "var_2": (ANY,),
                "var_3": (ANY,),
                "var_4": (ANY,),
            }
        }

    CATEGORY = "Rvaged/Folder"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "join_vars"

    def join_vars(self, var_1, var_2=None, var_3=None, var_4=None):
        variables = [var_1, var_2, var_3, var_4]
        return (','.join([str(var) for var in variables if var is not None]),)

#---------------------------------------------------------------------------------------------------------------------#
#imported from ImageSaver
class SamplerSelector:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampler_name": (comfy.samplers.KSampler.SAMPLERS,)}}

    CATEGORY = "Rvaged/Selector"
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("sampler_name",)
    FUNCTION = "get_names"

    def get_names(self, sampler_name):
        return (sampler_name,)
 
 #---------------------------------------------------------------------------------------------------------------------#
class SchedulerSelector:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler_comfy": (SCHEDULERS_COMFY,),
                "scheduler_efficient": (SCHEDULERS_EFFICIENT,),
                "scheduler_impact": (SCHEDULERS_IMPACT,),
                }
            }

    CATEGORY = "Rvaged/Selector"
    RETURN_TYPES = (
        SCHEDULERS_COMFY,
        SCHEDULERS_EFFICIENT, 
        SCHEDULERS_IMPACT, 
        "STRING",)
    RETURN_NAMES = ("scheduler_comfy", "scheduler_efficient", "scheduler_impact", "scheduler_name")
    FUNCTION = "get_names"

    def get_names(self, scheduler_comfy, scheduler_efficient, scheduler_impact):
        return (scheduler_comfy, scheduler_efficient, scheduler_impact, scheduler_impact)

#---------------------------------------------------------------------------------------------------------------------#
#imported from ComfyRoll and used as template for the following
class ImageSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input": ("INT", {"default": 1, "min": 1, "max": 2}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),            
            }
        }

    CATEGORY = "Rvaged/Switches"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "img_switch"

    def img_switch(self, Input, image1=None, image2=None):
        
        if Input == 1:
            return (image1,)
        else:
            return (image2,)

#---------------------------------------------------------------------------------------------------------------------#
class IntegerSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input": ("INT", {"default": 1, "min": 1, "max": 2}),
            },
            "optional": {
                "int1": ("INT", {"forceInput": True}),
                "int2": ("INT", {"forceInput": True}),
            }
        }

    CATEGORY = "Rvaged/Switches"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "execute"

    def execute(self, Input, int1=None, int2=None):
        
        if Input == 1:
            return (int1,)
        else:
            return (int2,)
 
#---------------------------------------------------------------------------------------------------------------------# 
class MaskSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input": ("INT", {"default": 1, "min": 1, "max": 2}),
            },
            "optional": {
                "mask1": ("MASK", {"forceInput": True}),
                "mask2": ("MASK", {"forceInput": True}),
            }
        }

    CATEGORY = "Rvaged/Switches"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"

    def execute(self, Input, mask1=None, mask2=None):
        
        if Input == 1:
            return (mask1,)
        else:
            return (mask2,)

#---------------------------------------------------------------------------------------------------------------------#
class LatentInputSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input": ("INT", {"default": 1, "min": 1, "max": 2}),
            },
            "optional": {
                "latent1": ("LATENT",),
                "latent2": ("LATENT",)          
            }
        }

    CATEGORY = "Rvaged/Switches"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "switch"

    def switch(self, Input, latent1=None, latent2=None):
        if Input == 1:
            return (latent1,)
        else:
            return (latent2,)

#---------------------------------------------------------------------------------------------------------------------#
class ConditioningInputSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input": ("INT", {"default": 1, "min": 1, "max": 2}),
            },
            "optional": {
                "conditioning1": ("CONDITIONING",),
                "conditioning2": ("CONDITIONING",),        
            }
        }

    CATEGORY = "Rvaged/Switches"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "switch"

    def switch(self, Input, conditioning1=None, conditioning2=None):
        if Input == 1:
            return (conditioning1,)
        else:
            return (conditioning2,)

#---------------------------------------------------------------------------------------------------------------------#
class ClipInputSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input": ("INT", {"default": 1, "min": 1, "max": 2}),
            },
            "optional": {
                "clip1": ("CLIP",),
                "clip2": ("CLIP",),      
            }
        }
    CATEGORY = "Rvaged/Switches"
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "switch"

    def switch(self, Input, clip1=None, clip2=None):
        if Input == 1:
            return (clip1,)
        else:
            return (clip2,)

#---------------------------------------------------------------------------------------------------------------------#
class ModelInputSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input": ("INT", {"default": 1, "min": 1, "max": 2}),
            },
            "optional": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),   
            }
        }
    CATEGORY = "Rvaged/Switches"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "switch"

    def switch(self, Input, model1=None, model2=None):
        if Input == 1:
            return (model1,)
        else:
            return (model2,)

#---------------------------------------------------------------------------------------------------------------------#
class TextInputSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input": ("INT", {"default": 1, "min": 1, "max": 2}),
            },
            "optional": {
                "text1": ("STRING", {"forceInput": True}),
                "text2": ("STRING", {"forceInput": True}), 
            }
        }

    CATEGORY = "Rvaged/Switches"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "switch"

    def switch(self, Input, text1=None, text2=None,):
        if Input == 1:
            return (text1,)
        else:
            return (text2,)

#---------------------------------------------------------------------------------------------------------------------#
class VAEInputSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input": ("INT", {"default": 1, "min": 1, "max": 2}),            
            },
            "optional": {
                "VAE1": ("VAE", {"forceInput": True}),
                "VAE2": ("VAE", {"forceInput": True}),
            }
        }

    CATEGORY = "Rvaged/Switches"
    RETURN_TYPES = ("VAE",)   
    FUNCTION = "switch"

    def switch(self, Input, VAE1=None, VAE2=None,):
        if Input == 1:
            return (VAE1,)
        else:
            return (VAE2,)

#---------------------------------------------------------------------------------------------------------------------#
# IMAGES TO RGB (WAS Node)
class Images2RGB:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    CATEGORY = "Rvaged/Convert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "Images_to_RGB"

    def Images_to_RGB(self, images):

        if len(images) > 1:
            tensors = []
            for image in images:
                tensors.append(pil2tensor(tensor2pil(image).convert('RGB')))
            tensors = torch.cat(tensors, dim=0)
            return (tensors, )
        else:
            return (pil2tensor(tensor2pil(images).convert("RGB")), )

#---------------------------------------------------------------------------------------------------------------------#        
#from comfy essentials
class ImageList2Batch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    CATEGORY = "Rvaged/Convert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "ImageList_to_Batch"
    INPUT_IS_LIST = True

    def ImageList_to_Batch(self, images):
        shape = images[0].shape[1:3]
        out = []

        for i in range(len(images)):
            img = p(images[i])
            if images[i].shape[1:3] != shape:
                transforms = T.Compose([
                    T.CenterCrop(min(img.shape[2], img.shape[3])),
                    T.Resize((shape[0], shape[1]), interpolation=T.InterpolationMode.BICUBIC),
                ])
                img = transforms(img)
            out.append(pb(img))
            #image[i] = pb(transforms(img))

        out = torch.cat(out, dim=0)

        return (out,)        

#---------------------------------------------------------------------------------------------------------------------#
#from impact
class ImageBatch2List:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",), }}

    CATEGORY = "Rvaged/Convert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "ImageBatch_to_List"

    def ImageBatch_to_List(self, images):
        iimages = [images[i:i + 1, ...] for i in range(images.shape[0])]
        return (iimages, )
