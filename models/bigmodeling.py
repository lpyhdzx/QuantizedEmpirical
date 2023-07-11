# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Union
import tempfile
import gc
import math

import torch
import torch.nn as nn

from accelerate.hooks import (
    AlignDevicesHook,
    CpuOffload,
    UserCpuOffloadHook,
    add_hook_to_module,
    attach_align_device_hook,
    attach_align_device_hook_on_blocks,
)
from accelerate.utils import (
    OffloadedWeightsLoader,
    check_device_map,
    extract_submodules_state_dict,
    find_tied_parameters,
    get_balanced_memory,
    infer_auto_device_map,
    load_checkpoint_in_model,
    offload_state_dict,
    retie_parameters,
    load_state_dict,
    set_module_tensor_to_device,
    offload_weight,
    load_offloaded_weights,
)
from accelerate.utils.offload import offload_weight, save_offload_index
from accelerate.utils.versions import is_torch_version


@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].

    </Tip>
    """
    if not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError("Initializing empty weights to a meta device requires torch >= 1.9.0")
    with init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
        yield f


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
        include_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_on_device

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Liner(100, 100)  # on `cuda` device
    ```
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer):
        old_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def cpu_offload(
    model: nn.Module,
    execution_device: Optional[torch.device] = None,
    offload_buffers: bool = False,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    preload_module_classes: Optional[List[str]] = None,
):
    """
    Activates full CPU offload for a model. As a result, all parameters of the model will be offloaded and only one
    copy of the state dict of the model will be kept. During the forward pass, parameters will be extracted from that
    state dict and put on the execution device passed as they are needed, then offloaded again.

    Args:
        model (`torch.nn.Module`):
            The model to offload.
        execution_device (`torch.device`, *optional*):
            The device on which the forward pass of the model will be executed (should be a GPU). Will default to the
            model first parameter device.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            The state dict of the model that will be kept on CPU.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
    """
    if not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError("CPU offloading requires torch >= 1.9.0")
    if execution_device is None:
        execution_device = next(iter(model.parameters())).device
    if state_dict is None:
        state_dict = {n: p.to("cpu") for n, p in model.state_dict().items()}

    add_hook_to_module(model, AlignDevicesHook(io_same_device=True), append=True)
    attach_align_device_hook(
        model,
        execution_device=execution_device,
        offload=True,
        offload_buffers=offload_buffers,
        weights_map=state_dict,
        preload_module_classes=preload_module_classes,
    )

    return model


def cpu_offload_with_hook(
    model: torch.nn.Module,
    execution_device: Optional[Union[int, str, torch.device]] = None,
    prev_module_hook: Optional[UserCpuOffloadHook] = None,
):
    """
    Offloads a model on the CPU and puts it back to an execution device when executed. The difference with
    [`cpu_offload`] is that the model stays on the execution device after the forward and is only offloaded again when
    the `offload` method of the returned `hook` is called. Useful for pipelines running a model in a loop.

    Args:
        model (`torch.nn.Module`):
            The model to offload.
        execution_device(`str`, `int` or `torch.device`, *optional*):
            The device on which the model should be executed. Will default to the MPS device if it's available, then
            GPU 0 if there is a GPU, and finally to the CPU.
        prev_module_hook (`UserCpuOffloadHook`, *optional*):
            The hook sent back by this function for a previous model in the pipeline you are running. If passed, its
            offload method will be called just before the forward of the model to which this hook is attached.

    Example:

    ```py
    model_1, hook_1 = cpu_offload_with_hook(model_1, cuda_device)
    model_2, hook_2 = cpu_offload_with_hook(model_2, cuda_device, prev_module_hook=hook_1)
    model_3, hook_3 = cpu_offload_with_hook(model_3, cuda_device, prev_module_hook=hook_2)

    hid_1 = model_1(input)
    for i in range(50):
        # model1 is offloaded on the CPU at the first iteration, model 2 stays on the GPU for this whole loop.
        hid_2 = model_2(hid_1)
    # model2 is offloaded to the CPU just before this forward.
    hid_3 = model_3(hid_3)

    # For model3, you need to manually call the hook offload method.
    hook_3.offload()
    ```
    """
    hook = CpuOffload(execution_device=execution_device, prev_module_hook=prev_module_hook)
    add_hook_to_module(model, hook, append=True)
    user_hook = UserCpuOffloadHook(model, hook)
    return model, user_hook


def disk_offload(
    model: nn.Module,
    offload_dir: Union[str, os.PathLike],
    execution_device: Optional[torch.device] = None,
    offload_buffers: bool = False,
    preload_module_classes: Optional[List[str]] = None,
):
    """
    Activates full disk offload for a model. As a result, all parameters of the model will be offloaded as
    memory-mapped array in a given folder. During the forward pass, parameters will be accessed from that folder and
    put on the execution device passed as they are needed, then offloaded again.

    Args:
        model (`torch.nn.Module`): The model to offload.
        offload_dir (`str` or `os.PathLike`):
            The folder in which to offload the model weights (or where the model weights are already offloaded).
        execution_device (`torch.device`, *optional*):
            The device on which the forward pass of the model will be executed (should be a GPU). Will default to the
            model's first parameter device.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
    """
    if not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError("Disk offloading requires torch >= 1.9.0")
    if not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, "index.json")):
        offload_state_dict(offload_dir, model.state_dict())
    if execution_device is None:
        execution_device = next(iter(model.parameters())).device
    weights_map = OffloadedWeightsLoader(save_folder=offload_dir)

    add_hook_to_module(model, AlignDevicesHook(io_same_device=True), append=True)
    attach_align_device_hook(
        model,
        execution_device=execution_device,
        offload=True,
        offload_buffers=offload_buffers,
        weights_map=weights_map,
        preload_module_classes=preload_module_classes,
    )

    return model


def dispatch_model(
    model: nn.Module,
    device_map: Dict[str, Union[str, int, torch.device]],
    main_device: Optional[torch.device] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    offload_dir: Optional[Union[str, os.PathLike]] = None,
    offload_index: Optional[Dict[str, str]] = None,
    offload_buffers: bool = False,
    preload_module_classes: Optional[List[str]] = None,
):
    """
    Dispatches a model according to a given device map. Layers of the model might be spread across GPUs, offloaded on
    the CPU or even the disk.

    Args:
        model (`torch.nn.Module`):
            The model to dispatch.
        device_map (`Dict[str, Union[str, int, torch.device]]`):
            A dictionary mapping module names in the models `state_dict` to the device they should go to. Note that
            `"disk"` is accepted even if it's not a proper value for `torch.device`.
        main_device (`str`, `int` or `torch.device`, *optional*):
            The main execution device. Will default to the first device in the `device_map` different from `"cpu"` or
            `"disk"`.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            The state dict of the part of the model that will be kept on CPU.
        offload_dir (`str` or `os.PathLike`):
            The folder in which to offload the model weights (or where the model weights are already offloaded).
        offload_index (`Dict`, *optional*):
            A dictionary from weight name to their information (`dtype`/ `shape` or safetensors filename). Will default
            to the index saved in `save_folder`.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
    """
    if not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError("Model dispatching requires torch >= 1.9.0")
    # Error early if the device map is incomplete.
    check_device_map(model, device_map)

    if main_device is None:
        if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {"cpu", "disk"}:
            main_device = "cpu"
        else:
            main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    if main_device != "cpu":
        cpu_modules = [name for name, device in device_map.items() if device == "cpu"]
        if state_dict is None and len(cpu_modules) > 0:
            state_dict = extract_submodules_state_dict(model.state_dict(), cpu_modules)

    disk_modules = [name for name, device in device_map.items() if device == "disk"]
    if offload_dir is None and offload_index is None and len(disk_modules) > 0:
        raise ValueError(
            "We need an `offload_dir` to dispatch this model according to this `device_map`, the following submodules "
            f"need to be offloaded: {', '.join(disk_modules)}."
        )
    if (
        len(disk_modules) > 0
        and offload_index is None
        and (not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, "index.json")))
    ):
        disk_state_dict = extract_submodules_state_dict(model.state_dict(), disk_modules)
        offload_state_dict(offload_dir, disk_state_dict)

    execution_device = {
        name: main_device if device in ["cpu", "disk"] else device for name, device in device_map.items()
    }
    execution_device[""] = main_device
    offloaded_devices = ["disk"] if main_device == "cpu" else ["cpu", "disk"]
    offload = {name: device in offloaded_devices for name, device in device_map.items()}
    save_folder = offload_dir if len(disk_modules) > 0 else None
    if state_dict is not None or save_folder is not None or offload_index is not None:
        device = main_device if offload_index is not None else None
        weights_map = OffloadedWeightsLoader(
            state_dict=state_dict, save_folder=save_folder, index=offload_index, device=device
        )
    else:
        weights_map = None

    tied_params = find_tied_parameters(model)
    attach_align_device_hook_on_blocks(
        model,
        execution_device=execution_device,
        offload=offload,
        offload_buffers=offload_buffers,
        weights_map=weights_map,
        preload_module_classes=preload_module_classes,
    )
    # Attaching the hook may break tied weights, so we retie them
    retie_parameters(model, tied_params)
    model.hf_device_map = device_map
    return model


def load_checkpoint_and_dispatch(
    model: nn.Module,
    checkpoint: Union[str, os.PathLike],
    device_map: Optional[Union[str, Dict[str, Union[int, str, torch.device]]]] = None,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[List[str]] = None,
    offload_folder: Optional[Union[str, os.PathLike]] = None,
    offload_buffers: bool = False,
    dtype: Optional[Union[str, torch.dtype]] = None,
    offload_state_dict: Optional[bool] = None,
    preload_module_classes: Optional[List[str]] = None,
):
    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded and adds the various hooks that will make this model run properly (even if split across devices).

    Args:
        model (`torch.nn.Module`): The model in which we want to load a checkpoint.
        checkpoint (`str` or `os.PathLike`):
            The folder checkpoint to load. It can be:
            - a path to a file containing a whole model state dict
            - a path to a `.json` file containing the index to a sharded checkpoint
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.

            To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For more
            information about each option see [here](big_modeling#designing-a-device-map).
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each GPU
            and the available CPU RAM if unset.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        offload_state_dict (`bool`, *optional*):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit. Will default to `True` if the device map
            picked contains `"disk"` values.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.

    Example:

    ```python
    >>> from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    >>> from huggingface_hub import hf_hub_download
    >>> from transformers import AutoConfig, AutoModelForCausalLM

    >>> # Download the Weights
    >>> checkpoint = "EleutherAI/gpt-j-6B"
    >>> weights_location = hf_hub_download(checkpoint, "pytorch_model.bin")

    >>> # Create a model and initialize it with empty weights
    >>> config = AutoConfig.from_pretrained(checkpoint)
    >>> with init_empty_weights():
    ...     model = AutoModelForCausalLM.from_config(config)

    >>> # Load the checkpoint and dispatch it to the right devices
    >>> model = load_checkpoint_and_dispatch(
    ...     model, weights_location, device_map="auto", no_split_module_classes=["GPTJBlock"]
    ... )
    ```
    """
    if not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError("Loading and dispatching requires torch >= 1.9.0")
    if isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
        raise ValueError(
            "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
            "'sequential'."
        )
    if device_map != "sequential":
        max_memory = get_balanced_memory(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes,
            dtype=dtype,
            low_zero=(device_map == "balanced_low_0"),
        )
    if isinstance(device_map, str):
        device_map = infer_auto_device_map(
            model, max_memory=max_memory, no_split_module_classes=no_split_module_classes, dtype=dtype
        )
    if offload_state_dict is None and device_map is not None and "disk" in device_map.values():
        offload_state_dict = True
    load_checkpoint_in_model(
        model,
        checkpoint,
        device_map=device_map,
        offload_folder=offload_folder,
        dtype=dtype,
        offload_state_dict=offload_state_dict,
        offload_buffers=offload_buffers,
    )
    if device_map is None:
        return model
    return dispatch_model(
        model,
        device_map=device_map,
        offload_dir=offload_folder,
        offload_buffers=offload_buffers,
        preload_module_classes=preload_module_classes,
    )

def load_checkpoint_in_model(
    model: nn.Module,
    checkpoint: Union[str, os.PathLike],
    device_map: Optional[Dict[str, Union[int, str, torch.device]]] = None,
    offload_folder: Optional[Union[str, os.PathLike]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    offload_state_dict: bool = False,
    offload_buffers: bool = False,
):
    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded.

    <Tip warning={true}>

    Once loaded across devices, you still need to call [`dispatch_model`] on your model to make it able to run. To
    group the checkpoint loading and dispatch in one single call, use [`load_checkpoint_and_dispatch`].

    </Tip>

    Args:
        model (`torch.nn.Module`):
            The model in which we want to load a checkpoint.
        checkpoint (`str` or `os.PathLike`):
            The folder checkpoint to load. It can be:
            - a path to a file containing a whole model state dict
            - a path to a `.json` file containing the index to a sharded checkpoint
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        offload_state_dict (`bool`, *optional*, defaults to `False`):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit.
        offload_buffers (`bool`, *optional*, defaults to `False):
            Whether or not to include the buffers in the weights offloaded to disk.
    """
    tied_params = find_tied_parameters(model)
    if offload_folder is None and device_map is not None and "disk" in device_map.values():
        raise ValueError(
            "At least one of the model submodule will be offloaded to disk, please pass along an `offload_folder`."
        )
    elif offload_folder is not None and device_map is not None and "disk" in device_map.values():
        os.makedirs(offload_folder, exist_ok=True)

    if isinstance(dtype, str):
        # We accept "torch.float16" or just "float16"
        dtype = dtype.replace("torch.", "")
        dtype = getattr(torch, dtype)

    checkpoint_files = None
    index_filename = None
    if os.path.isfile(checkpoint):
        if str(checkpoint).endswith(".json"):
            index_filename = checkpoint
        else:
            checkpoint_files = [checkpoint]
    elif os.path.isdir(checkpoint):
        potential_index = [f for f in os.listdir(checkpoint) if f.endswith(".index.json")]
        if len(potential_index) == 0:
            raise ValueError(f"{checkpoint} is not a folder containing a `.index.json` file.")
        elif len(potential_index) == 1:
            index_filename = os.path.join(checkpoint, potential_index[0])
        else:
            raise ValueError(f"{checkpoint} containing more than one `.index.json` file, delete the irrelevant ones.")
    else:
        raise ValueError(
            "`checkpoint` should be the path to a file containing a whole state dict, or the index of a sharded "
            f"checkpoint, or a folder containing a sharded checkpoint, but got {checkpoint}."
        )

    if index_filename is not None:
        checkpoint_folder = os.path.split(index_filename)[0]
        with open(index_filename, "r") as f:
            index = json.loads(f.read())

        if "weight_map" in index:
            index = index["weight_map"]
        checkpoint_files = sorted(list(set(index.values())))
        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]

    # Logic for missing/unexepected keys goes here.

    offload_index = {}
    if offload_state_dict:
        state_dict_folder = tempfile.mkdtemp()
        state_dict_index = {}

    buffer_names = [name for name, _ in model.named_buffers()]

    for checkpoint_file in checkpoint_files:
        checkpoint = load_state_dict(checkpoint_file, device_map=device_map)
        if device_map is None:
            model.load_state_dict(checkpoint, strict=False)
        else:
            for param_name, param in checkpoint.items():
                module_name = param_name

                while len(module_name) > 0 and module_name not in device_map:
                    module_name = ".".join(module_name.split(".")[:-1])
                if module_name == "" and "" not in device_map:
                    # TODO: group all errors and raise at the end.
                    raise ValueError(f"{param_name} doesn't have any device set.")
                param_device = device_map[module_name]

                if param_device == "disk":
                    if offload_buffers or param_name not in buffer_names:
                        set_module_tensor_to_device(model, param_name, "meta")
                    offload_weight(param, param_name, offload_folder, index=offload_index)
                elif param_device == "cpu" and offload_state_dict:
                    set_module_tensor_to_device(model, param_name, "meta")
                    offload_weight(param, param_name, state_dict_folder, index=state_dict_index)
                else:
                    set_module_tensor_to_device(model, param_name, param_device, value=param, dtype=dtype)
            print(device_map)
            for param_name, param in model.named_parameters():
                if not param.is_meta:
                    continue
                module_name = param_name

                while len(module_name) > 0 and module_name not in device_map:
                    module_name = ".".join(module_name.split(".")[:-1])
                if module_name == "" and "" not in device_map:
                    # TODO: group all errors and raise at the end.
                    raise ValueError(f"{param_name} doesn't have any device set.")
                param_device = device_map[module_name]

                if param_device == "disk":
                    if offload_buffers or param_name not in buffer_names:
                        set_module_tensor_to_device(model, param_name, "meta")
                    offload_weight(param, param_name, offload_folder, index=offload_index)
                elif param_device == "cpu" and offload_state_dict:
                    set_module_tensor_to_device(model, param_name, "meta")
                    offload_weight(param, param_name, state_dict_folder, index=state_dict_index)
                else:
                    if "lora_A" in param_name:
                        lora_A = nn.Linear(param.shape[1], param.shape[0], bias=False)
                        nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
                        param = lora_A.weight
                    elif "lora_B" in param_name:
                        lora_B = nn.Linear(param.shape[1], param.shape[0], bias=False)
                        nn.init.zeros_(lora_B.weight)
                        param = lora_B.weight
                    else:
                        print("param name", param_name)
                        raise NotImplementedError()
                    set_module_tensor_to_device(model, param_name, param_device, value=param, dtype=dtype)

        # Force Python to clean up.
        del checkpoint
        gc.collect()

    save_offload_index(offload_index, offload_folder)

    # Load back offloaded state dict on CPU
    if offload_state_dict:
        load_offloaded_weights(model, state_dict_index, state_dict_folder)
        shutil.rmtree(state_dict_folder)

    retie_parameters(model, tied_params)