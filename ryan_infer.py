import rp
# from rp import *
import torch
import numpy as np
import einops
from diffusers import CogVideoXImageToVideoPipeline
from diffusers import CogVideoXVideoToVideoPipeline
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video, load_image
from icecream import ic
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel 
from transformers import T5EncoderModel

import rp.git.CommonSource.noise_warp as nw

pipe_ids = dict(
    T2V5B="THUDM/CogVideoX-5b",
    T2V2B="THUDM/CogVideoX-2b",
    I2V5B="THUDM/CogVideoX-5b-I2V",
)

# From a bird's-eye view, a serene scene unfolds: a herd of deer gracefully navigates shallow, warm-hued waters, their silhouettes stark against the earthy tones. The deer, spread across the frame, cast elongated, well-defined shadows that accentuate their antlers, creating a mesmerizing play of light and dark. This aerial perspective captures the tranquil essence of the setting, emphasizing the harmonious contrast between the deer and their mirror-like reflections on the water's surface. The composition exudes a peaceful stillness, yet the subtle movement suggested by the shadows adds a dynamic layer to the natural beauty and symmetry of the moment.
base_url = 'https://huggingface.co/OneOverZero/Go-With-The-Flow/resolve/main/'
lora_urls = dict(
    I2V5B_final_i30000_lora_weights                          = base_url+'I2V5B_final_i30000_lora_weights.safetensors',
    I2V5B_final_i38800_nearest_lora_weights                  = base_url+'I2V5B_final_i38800_nearest_lora_weights.safetensors',
    I2V5B_resum_blendnorm_0degrad_i13600_webvid_lora_weights = base_url+'I2V5B_resum_blendnorm_0degrad_i13600_webvid_lora_weights.safetensors',
    T2V2B_RDeg_i30000_lora_weights                           = base_url+'T2V2B_RDeg_i30000_lora_weights.safetensors',
    T2V5B_blendnorm_i18000_envato_lora_weights               = base_url+'T2V5B_blendnorm_i18000_envato_lora_weights.safetensors',
    T2V5B_blendnorm_i25000_envato_nearest_lora_weights       = base_url+'T2V5B_blendnorm_i25000_envato_nearest_lora_weights.safetensors',
)

dtype=torch.bfloat16

#https://medium.com/@ChatGLM/open-sourcing-cogvideox-a-step-towards-revolutionizing-video-generation-28fa4812699d
B, F, C, H, W = 1, 13, 16, 60, 90  # The defaults
num_frames=(F-1)*4+1 #https://miro.medium.com/v2/resize:fit:1400/format:webp/0*zxsAG1xks9pFIsoM
#Possible num_frames: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49
assert num_frames==49

@rp.memoized #Torch never manages to unload it from memory anyway
def get_pipe(model_name, device=None, low_vram=True):
    """
    model_name is like "I2V5B", "T2V2B", or "T2V5B", or a LoRA name like "T2V2B_RDeg_i30000_lora_weights"
    device is automatically selected if unspecified
    low_vram, if True, will make the pipeline use CPU offloading
    """

    if model_name in pipe_ids:
        lora_name = None
        pipe_name = model_name
    else:
        #By convention, we have lora_paths that start with the pipe names
        rp.fansi_print(f"Getting pipe name from model_name={model_name}",'cyan','bold')
        lora_name = model_name
        pipe_name = lora_name.split('_')[0]

    is_i2v = "I2V" in pipe_name  # This is a convention I'm using right now
    # is_v2v = "V2V" in pipe_name  # This is a convention I'm using right now

    # if is_v2v:
    #     old_pipe_name = pipe_name
    #     old_lora_name = lora_name
    #     if pipe_name is not None: pipe_name = pipe_name.replace('V2V','T2V')
    #     if lora_name is not None: lora_name = lora_name.replace('V2V','T2V')
    #     rp.fansi_print(f"V2V: {old_pipe_name} --> {pipe_name}   &&&   {old_lora_name} --> {lora_name}",'white','bold italic','red')

    pipe_id = pipe_ids[pipe_name]
    print(f"LOADING PIPE WITH device={device} pipe_name={pipe_name} pipe_id={pipe_id} lora_name={lora_name}" )
    
    hub_model_id = pipe_ids[pipe_name]

    transformer = CogVideoXTransformer3DModel.from_pretrained(hub_model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    text_encoder = T5EncoderModel.from_pretrained(hub_model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLCogVideoX.from_pretrained(hub_model_id, subfolder="vae", torch_dtype=torch.bfloat16)

    PipeClass = CogVideoXImageToVideoPipeline if is_i2v else CogVideoXPipeline
    pipe = PipeClass.from_pretrained(hub_model_id, torch_dtype=torch.bfloat16, vae=vae,transformer=transformer,text_encoder=text_encoder)

    if lora_name is not None:
        lora_folder = rp.make_directory('lora_models')
        lora_url = lora_urls[lora_name]
        lora_path = rp.download_url(lora_url, lora_folder, show_progress=True, skip_existing=True)
        assert rp.file_exists(lora_path), (lora_name, lora_path)
        print(end="\tLOADING LORA WEIGHTS...",flush=True)
        pipe.load_lora_weights(lora_path)
        print("DONE!")

    if device is None:
        device = rp.select_torch_device()

    if not low_vram:
        print("\tUSING PIPE DEVICE", device)
        pipe = pipe.to(device)
    else:
        print("\tUSING PIPE DEVICE WITH CPU OFFLOADING",device)
        pipe=pipe.to('cpu')
        pipe.enable_sequential_cpu_offload(device=device)

    # pipe.vae.enable_tiling()
    # pipe.vae.enable_slicing()

    # Metadata
    pipe.lora_name = lora_name
    pipe.pipe_name = pipe_name
    pipe.is_i2v    = is_i2v
    # pipe.is_v2v    = is_v2v
    
    return pipe

def get_downtemp_noise(noise, noise_downtemp_interp):
    assert noise_downtemp_interp in {'nearest', 'blend', 'blend_norm', 'randn'}, noise_downtemp_interp
    if   noise_downtemp_interp == 'nearest'    : return                  rp.resize_list(noise, 13)
    elif noise_downtemp_interp == 'blend'      : return                   downsamp_mean(noise, 13)
    elif noise_downtemp_interp == 'blend_norm' : return normalized_noises(downsamp_mean(noise, 13))
    elif noise_downtemp_interp == 'randn'      : return torch.randn_like(rp.resize_list(noise, 13)) #Basically no warped noise, just r
    else: assert False, 'impossible'

def downsamp_mean(x, l=13):
    return torch.stack([rp.mean(u) for u in rp.split_into_n_sublists(x, l)])

def normalized_noises(noises):
    #Noises is in TCHW form
    return torch.stack([x / x.std(1, keepdim=True) for x in noises])


@rp.memoized
def load_sample_cartridge(
    sample_path: str,
    degradation=0,
    noise_downtemp_interp='nearest',
    image=None,
    prompt=None,
    #SETTINGS:
    num_inference_steps=30,
    guidance_scale=6,
):
    """
    COMPLETELY FROM SAMPLE: Generate with /root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidSampleGenerator.ipynb
    EXAMPLE PATHS:
        sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/plus_pug.pkl'
        sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/amuse_chop.pkl'
        sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/chomp_shop.pkl'
        sample_path = '/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/ahead_job.pkl'
        sample_path = rp.random_element(glob.glob('/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/*.pkl'))
    """

    #These could be args in the future. I can't think of a use case yet though, so I'll keep the signature clean.
    noise=None
    video=None

    if rp.is_a_folder(sample_path):
        #Was generated using the flow pipeline
        print(end="LOADING CARTRIDGE FOLDER "+sample_path+"...")
        
        noise_file=rp.path_join(sample_path,'noises.npy')
        instance_noise = np.load(noise_file)
        instance_noise = torch.tensor(instance_noise)
        instance_noise = einops.rearrange(instance_noise, 'F H W C -> F C H W')

        video_file=rp.path_join(sample_path,'input.mp4')
        instance_video = rp.load_video(video_file)
        instance_video = rp.as_torch_images(instance_video)
        instance_video = instance_video * 2 - 1

        sample = rp.as_easydict(
            instance_prompt = '', #Please have some prompt to override this! Ideally the defualt would come from a VLM
            instance_noise = instance_noise,
            instance_video = instance_video,
        )

        print("DONE!")
        
    else:
        #Was generated using the Cut-And-Drag GUI
        print(end="LOADING CARTRIDGE FILE "+sample_path+"...")
        sample=rp.file_to_object(sample_path)
        print("DONE!")

    #SAMPLE EXAMPLE:
    #    >>> sample=file_to_object('/root/micromamba/envs/i2sb/lib/python3.8/site-packages/rp/git/CommonSource/notebooks/CogVidX_Saved_Train_Samples/ahead_job.pkl')
    #    >>> list(sample)?s                 -->  ['instance_prompt', 'instance_video', 'instance_noise']
    #    >>> sample.instance_prompt?s       -->  A group of elk, including a dominant bull, is seen grazing and moving through...
    #    >>> sample.instance_noise.shape?s  -->  torch.Size([49, 16,  60,  90])
    #    >>> sample.instance_video.shape?s  -->  torch.Size([49,  3, 480, 720])   # Range: [-1, 1]

    sample_noise  = sample["instance_noise" ].to(dtype)
    sample_video  = sample["instance_video" ].to(dtype)
    sample_prompt = sample["instance_prompt"]

    sample_gif_path = sample_path+'.mp4'
    if not rp.file_exists(sample_gif_path):
        sample_gif_path = sample_path+'.gif' #The older scripts made this. Backwards compatibility.
    if not rp.file_exists(sample_gif_path):
        #Create one!
        #Clientside warped noise does not come with a nice GIF so we make one here and now!
        sample_gif_path = sample_path+'.mp4'

        rp.fansi_print("MAKING SAMPLE PREVIEW VIDEO",'light blue green','underlined')
        preview_sample_video=rp.as_numpy_images(sample_video)/2+.5
        preview_sample_noise=rp.as_numpy_images(sample_noise)[:,:,:,:3]/5+.5
        preview_sample_noise = rp.resize_images(preview_sample_noise, size=8, interp="nearest")
        preview_sample=rp.horizontally_concatenated_videos(preview_sample_video,preview_sample_noise)
        rp.save_video_mp4(preview_sample,sample_gif_path,video_bitrate='max',framerate=12)
        rp.fansi_print("DONE MAKING SAMPLE PREVIEW VIDEO!",'light blue green','underlined')

    #prompt=sample.instance_prompt
    downtemp_noise = get_downtemp_noise(
        sample_noise,
        noise_downtemp_interp=noise_downtemp_interp,
    )
    downtemp_noise = downtemp_noise[None]
    downtemp_noise = nw.mix_new_noise(downtemp_noise, degradation)

    assert downtemp_noise.shape == (B, F, C, H, W), (downtemp_noise.shape,(B, F, C, H, W))

    if image is None            : sample_image = rp.as_pil_image(rp.as_numpy_image(sample_video[0].float()/2+.5))
    elif isinstance(image, str) : sample_image = rp.as_pil_image(rp.as_rgb_image(rp.load_image(image)))
    else                        : sample_image = rp.as_pil_image(rp.as_rgb_image(image))

    metadata = rp.gather_vars('sample_path degradation downtemp_noise sample_gif_path sample_video sample_noise noise_downtemp_interp')
    settings = rp.gather_vars('num_inference_steps guidance_scale'+0*'v2v_strength')

    if noise  is None: noise  = downtemp_noise
    if video  is None: video  = sample_video
    if image  is None: image  = sample_image
    if prompt is None: prompt = sample_prompt

    assert noise.shape == (B, F, C, H, W), (noise.shape,(B, F, C, H, W))

    return rp.gather_vars('prompt noise image video metadata settings')

def dict_to_name(d=None, **kwargs):
    """
    Used to generate MP4 file names
    
    EXAMPLE:
        >>> dict_to_name(dict(a=5,b='hello',c=None))
        ans = a=5,b=hello,c=None
        >>> name_to_dict(ans)
        ans = {'a': '5', 'b': 'hello', 'c': 'None'}
    """
    if d is None:
        d = {}
    d.update(kwargs)
    return ",".join("=".join(map(str, [key, value])) for key, value in d.items())

# def name_to_dict(nam"
#     Useful for analyzing output MP4 files
#
#     EXAMPLE:
#         >>> dict_to_name(dict(a=5,b='hello',c=None))
#         ans = a=5,b=hello,c=None
#         >>> name_to_dict(ans)
#         ans = {'a': '5', 'b': 'hello', 'c': 'None'}
#     """
#     output=rp.as_easydict()
#     for entry in name.split(','):
#         key,value=entry.split('=',maxsplit=1)
#         output[key]=value
#     return output
#
#
def get_output_path(pipe, cartridge, subfolder:str, output_root:str):
    """
    Generates a unique output path for saving a generated video.

    Args:
        pipe: The video generation pipeline used.
        cartridge: Data used for generating the video.
        subfolder (str): Subfolder for saving the video.
        output_root (str): Root directory for output videos.

    Returns:
        String representing the unique path to save the video.
    """

    time = rp.millis()

    output_name = (
        dict_to_name(
            t=time,
            pipe=pipe.pipe_name,
            lora=pipe.lora_name,
            steps    =               cartridge.settings.num_inference_steps,
            # strength =               cartridge.settings.v2v_strength,
            degrad   =               cartridge.metadata.degradation,
            downtemp =               cartridge.metadata.noise_downtemp_interp,
            samp     = rp.get_file_name(rp.get_parent_folder(cartridge.metadata.sample_path), False),
        )
        + ".mp4"
    )

    output_path = rp.get_unique_copy_path(
        rp.path_join(
            rp.make_directory(
                rp.path_join(output_root, subfolder),
            ),
            output_name,
        ),
    )

    rp.fansi_print(f"OUTPUT PATH: {rp.fansi_highlight_path(output_path)}", "blue", "bold")

    return output_path

def run_pipe(
    pipe,
    cartridge,
    subfolder="first_subfolder",
    output_root: str = "infer_outputs",
    output_mp4_path = None, #This overrides subfolder and output_root if specified
):
    # output_mp4_path = output_mp4_path or get_output_path(pipe, cartridge, subfolder, output_root)

    if rp.file_exists(output_mp4_path):
        raise RuntimeError("{output_mp4_path} already exists! Please choose a different output file or delete that one. This script is designed not to clobber previous results.")
    
    if pipe.is_i2v:
        image = cartridge.image
        if isinstance(image, str):
            image = rp.load_image(image,use_cache=True)
        image = rp.as_pil_image(rp.as_rgb_image(image))

    # if pipe.is_v2v:
    #     print("Making v2v video...")
    #     v2v_video=cartridge.video
    #     v2v_video=rp.as_numpy_images(v2v_video) / 2 + .5
    #     v2v_video=rp.as_pil_images(v2v_video)

    print("NOISE SHAPE",cartridge.noise.shape)
    print("IMAGE",image)

    video = pipe(
        prompt=cartridge.prompt,
        **(dict(image   =image                          ) if pipe.is_i2v else {}),
        # **(dict(strength=cartridge.settings.v2v_strength) if pipe.is_v2v else {}),
        # **(dict(video   =v2v_video                      ) if pipe.is_v2v else {}),
        num_inference_steps=cartridge.settings.num_inference_steps,
        latents=cartridge.noise,

        guidance_scale=cartridge.settings.guidance_scale,
        # generator=torch.Generator(device=device).manual_seed(42),
    ).frames[0]

    export_to_video(video, output_mp4_path, fps=8)

    sample_gif=rp.load_video(cartridge.metadata.sample_gif_path)
    video=rp.as_numpy_images(video)
    prevideo = rp.horizontally_concatenated_videos(
        rp.resize_list(sample_gif, len(video)),
        video,
        origin='bottom right',
    )
    import textwrap
    prevideo = rp.labeled_images(
        prevideo,
        position="top",
        labels=cartridge.metadata.sample_path +"\n"+output_mp4_path +"\n\n" + rp.wrap_string_to_width(cartridge.prompt, 250),
        size_by_lines=True,
        text_color='light light light blue',
        # font='G:Lexend'
    )

    preview_mp4_path = output_mp4_path + "_preview.mp4"
    preview_gif_path = preview_mp4_path + ".gif"
    print(end=f"Saving preview MP4 to preview_mp4_path = {preview_mp4_path}...")
    rp.save_video_mp4(prevideo, preview_mp4_path, framerate=16, video_bitrate="max", show_progress=False)
    compressed_preview_mp4_path = rp.save_video_mp4(prevideo, output_mp4_path + "_preview_compressed.mp4", framerate=16, show_progress=False)
    print("done!")
    print(end=f"Saving preview gif to preview_gif_path = {preview_gif_path}...")
    rp.convert_to_gif_via_ffmpeg(preview_mp4_path, preview_gif_path, framerate=12,show_progress=False)
    print("done!")

    return rp.gather_vars('video output_mp4_path preview_mp4_path compressed_preview_mp4_path cartridge subfolder preview_mp4_path preview_gif_path')


# #prompt = "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
# prompt = "An old house by the lake with wooden plank siding and a thatched roof"
# prompt = "Soaring through deep space"
# prompt = "Swimming by the ruins of the titanic"
# prompt = "A camera flyby of a gigantic ice tower that a princess lives in, zooming in from far away from the castle into her dancing in the window"
# prompt = "A drone flyby of the grand canyon, aerial view"
# prompt = "A bunch of puppies running around a front lawn in a giant courtyard "
# #image = load_image(image=download_url_to_cache("https://media.sciencephoto.com/f0/22/69/89/f0226989-800px-wm.jpg"))

def main(
    sample_path,
    output_mp4_path:str,
    prompt=None,
    degradation=.5,
    model_name='I2V5B_final_i38800_nearest_lora_weights',

    low_vram=True,
    device:str=None,
    
    #BROADCASTABLE:
    noise_downtemp_interp='nearest',
    image=None,
    num_inference_steps=30,
    guidance_scale=6,
    # v2v_strength=.5,#Timestep for when using Vid2Vid. Only set to not none when using a T2V model!
):
    """
    Main function to run the video generation pipeline with specified parameters.

    Args:
        model_name (str): Name of the pipeline to use ('T2V5B', 'T2V2B', 'I2V5B', etc).
        device (str or int, optional): Device to run the model on (e.g., 'cuda:0' or 0). If unspecified, the GPU with the  most free VRAM will be chosen.
        low_vram (bool): Set to True if you have less than 32GB of VRAM. In enables model cpu offloading, which slows down inference but needs much less vram.
        sample_path (str or list, optional): Broadcastable. Path(s) to the sample `.pkl` file(s) or folders containing (noise.npy and input.mp4 files)
        degradation (float or list): Broadcastable. Degradation level(s) for the noise warp (float between 0 and 1).
        noise_downtemp_interp (str or list): Broadcastable. Interpolation method(s) for down-temporal noise. Options: 'nearest', 'blend', 'blend_norm'.
        image (str, PIL.Image, or list, optional): Broadcastable. Image(s) to use as the initial frame(s). Can be a URL or a path to an image.
        prompt (str or list, optional): Broadcastable. Text prompt(s) for video generation.
        num_inference_steps (int or list): Broadcastable. Number of inference steps for the pipeline.
    """
    output_root='infer_outputs', # output_root (str): Root directory where output videos will be saved.
    subfolder='default_subfolder', # subfolder (str): Subfolder within output_root to save outputs.

    if device is None:
        device = rp.select_torch_device(reserve=True, prefer_used=True)
        rp.fansi_print(f"Selected torch device: {device}")


    cartridge_kwargs = rp.broadcast_kwargs(
        rp.gather_vars(
            "sample_path",
            "degradation",
            "noise_downtemp_interp",
            "image",
            "prompt",
            "num_inference_steps",
            "guidance_scale",
            # "v2v_strength",
        )
    )

    rp.fansi_print("cartridge_kwargs:", "cyan", "bold")
    print(
        rp.indentify(
            rp.with_line_numbers(
                rp.fansi_pygments(
                    rp.autoformat_json(cartridge_kwargs),
                    "json",
                ),
                align=True,
            )
        ),
    )

    # cartridges = [load_sample_cartridge(**x) for x in cartridge_kwargs]
    cartridges = rp.load_files(lambda x:load_sample_cartridge(**x), cartridge_kwargs, show_progress='eta:Loading Cartridges')

    pipe = get_pipe(model_name, device, low_vram=low_vram)

    output=[]
    for cartridge in cartridges:
        pipe_out = run_pipe(
            pipe=pipe,
            cartridge=cartridge,
            output_root=output_root,
            subfolder=subfolder,
            output_mp4_path=output_mp4_path,
        )

        output.append(
            rp.as_easydict(
                rp.gather(
                    pipe_out,
                    [
                        "output_mp4_path",
                        "preview_mp4_path",
                        "compressed_preview_mp4_path",
                        "preview_mp4_path",
                        "preview_gif_path",
                    ],
                    as_dict=True,
                )
            )
        )
    return output

if __name__ == '__main__':
    import fire
    fire.Fire(main)



