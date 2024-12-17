To run this code:

For cut-and-drag motion control:

Locally:
- Run ```pip install -r requirements.txt```
- Then run ```python gui.py```
- Follow the instructions given to you on that GUI.
- After completion, it will generate an MP4 file. Upload that file to a computer with decent GPU's.

On the computer with a decent GPU (24GB or more VRAM):
- Run ```pip install -r requirements.txt```
- Warp the noise. Run ```PY make_warped_noise.py <PATH TO VIDEO OR URL> --ouptut_folder noise_warp_output_folder```
- Then run ```ryan_infer.py noise_warp_output_folder --prompt "A duck splashing" --output_mp4_path 'output.mp4' --device 'cuda' --num_inference_steps 5```
- Replace folder names and hyperparameters as you see fit! The output will be ```output.mp4```
