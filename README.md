<!-- # Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise -->

<p align="center">
  <img src="assets/Logo.png" alt="Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise" width="100%">
</p>

[**Project Page**](https://gowiththeflowpaper.github.io)

---

## Abstract

Go-with-the-Flow is an easy and efficient way to control the motion patterns of video diffusion models. It lets a user decide how the camera and objects in a scene will move, and can even let you transfer motion patterns from one video to another.

We simply fine-tune a base model — requiring no changes to the original pipeline or architecture, except: instead of using pure i.i.d. Gaussian noise, we use **warped noise** instead. Inference has exactly the same computational cost as running the base model.

---

## Quick Start: Cut-and-drag Motion Control

Cut-and-drag motion control lets you take an image, and create a video by cutting out different parts of that image and dragging them around.

For cut-and-drag motion control, there are two parts: an GUI to create a crude animation (no GPU needed), then a diffusion script to turn that crude animation into a pretty one (requires GPU).

**YouTube Tutorial**: [https://www.youtube.com/watch?v=lt16s6tFOnI](https://www.youtube.com/watch?v=lt16s6tFOnI)

Examples:

<p align="center">
  <img src="assets/cut_and_drag_example_1.gif" width="450">
  <img src="assets/cut_and_drag_example_2.gif" width="450">
  <img src="assets/cut_and_drag_example_3.gif" width="450">
  <img src="assets/cut_and_drag_example_4.gif" width="450">
  <img src="assets/cut_and_drag_example_5.gif" width="450">
</p>

### 1. Animation Template GUI (Local)

1. Clone this repo, then `cd` into it.  
2. Install local requirements:

    `pip install -r requirements_local.txt`

3. Run the GUI:

    `python cut_and_drag_gui.py`

4. Follow the instructions shown in the GUI.  

After completion, an MP4 file will be generated. You’ll need to move this file to a computer with a decent GPU to continue.

### 2. Running Video Diffusion (GPU)

1. Clone this repo on the machine with the GPU, then `cd` into it.  
2. Install requirements:

    `pip install -r requirements.txt`

3. Warp the noise (replace `<PATH TO VIDEO OR URL>` accordingly):

    `PY make_warped_noise.py <PATH TO VIDEO OR URL> --ouptut_folder noise_warp_output_folder`

4. Run inference:

    ```
    python cut_and_drag_inference.py noise_warp_output_folder \
        --prompt "A duck splashing" \
        --output_mp4_path "output.mp4" \
        --device "cuda" \
        --num_inference_steps 5
    ```

Adjust folder paths, prompts, and other hyperparameters as needed. The output will be saved as `output.mp4`.

---

## Citation

If you use this in your research, please consider citing:

    @misc{gowiththeflow2023,
      title={Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise},
      author={...},
      year={2023},
      howpublished={\url{https://gowiththeflowpaper.github.io}},
    }

---

## License

This project is released under the [LICENSE](LICENSE) of your choice.

---

Thanks for checking out **Go-with-the-Flow**! For questions or feedback, feel free to open an issue or contact us directly.
