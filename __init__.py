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

from .nodes import *

NODE_CLASS_MAPPINGS = {
    #conversion
    "Integer to String": Int2Str,
    "Float to String": Float2Str,
    "Float to Integer": Float2Int,
    "Images2RGB": Images2RGB,
    "Imagelist2Batch": ImageList2Batch, 
    "ImageBatch2List": ImageBatch2List,
    "Masklist2Batch": MaskList2Batch, 
    "MaskBatch2List": MaskBatch2List,
    #folder
    "Create Project Folder": CreateProjectFolder,
    "Add Folder": Add_Folder,
    "Add File Name Prefix": Add_FileNamePrefix,
    #operations
    "If ANY return A else B": IfExecute,
    "Replace String": ReplaceString,
    "Concat String": MergeString,
    "Join Variables": Join_Vars,
    "Join Variables V2": Join_Vars_V2,
    #passer
    "Pass Clip": PassClip,
    "Pass Images": PassImages,
    "Pass Latent": PassLatent,
    "Pass Masks": PassMasks,
    "Pass Model": PassModel,
    #primitives
    "Boolean": CBoolean, 
    "Float": CFloat, 
    "Integer": CInteger, 
    "String": CText, 
    "String (Multiline)": CTextML, 
    #selector
    "Sampler Selector": SamplerSelector, 
    "Sampler Selector (Restart)": SamplerSelectorRestart, 
    "Scheduler Selector": SchedulerSelector, 
    "Scheduler Selector (ComfyUI)": SchedulerSelectorComfyUI,
    "Scheduler Selector (Efficient)": SchedulerSelectorEfficient,
    "Scheduler Selector (Impact)": SchedulerSelectorImpact,
    "Scheduler Selector (Restart)": SchedulerSelectorRestart,
    #switches
    "Audio Switch": AUDIOInputSwitch, 
    "Image Switch": ImageSwitch, 
    "Integer Switch": IntegerSwitch, 
    "Mask Switch": MaskSwitch, 
    "Latent Switch": LatentInputSwitch, 
    "Conditioning Switch": ConditioningInputSwitch, 
    "Clip Switch": ClipInputSwitch, 
    "Model Switch": ModelInputSwitch, 
    "Text Switch": TextInputSwitch, 
    "VAE Switch": VAEInputSwitch, 
    "IntValueGrp": IntValueGrp, 
}
