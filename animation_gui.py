from rp import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon as Polygon
import cv2
git_import('CommonSource')
import rp.git.CommonSource.noise_warp as nw


def select_polygon(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Left click to add points. Right click to undo. Close the window to finish.")

    path = []

    def onclick(event):
        if event.button == 1:  # Left click
            if event.xdata is not None and event.ydata is not None:
                path.append((event.xdata, event.ydata))
            ax.clear()
            ax.imshow(image)
            ax.set_title("Left click to add points. Right click to undo. Close the window to finish.")
            for i in range(len(path)):
                if i > 0:
                    ax.plot([path[i - 1][0], path[i][0]], [path[i - 1][1], path[i][1]], "r-")
                ax.plot(path[i][0], path[i][1], "ro")
            if len(path) > 1:
                ax.plot([path[-1][0], path[0][0]], [path[-1][1], path[0][1]], "r--")
            if len(path) > 2:
                polygon = Polygon(path, closed=True, alpha=0.3, facecolor="r", edgecolor="r")
                ax.add_patch(polygon)
            fig.canvas.draw()
        elif event.button == 3 and path:  # Right click
            path.pop()
            ax.clear()
            ax.imshow(image)
            ax.set_title("Left click to add points. Right click to undo. Close the window to finish.")
            for i in range(len(path)):
                if i > 0:
                    ax.plot([path[i - 1][0], path[i][0]], [path[i - 1][1], path[i][1]], "r-")
                ax.plot(path[i][0], path[i][1], "ro")
            if len(path) > 1:
                ax.plot([path[-1][0], path[0][0]], [path[-1][1], path[0][1]], "r--")
            if len(path) > 2:
                polygon = Polygon(path, closed=True, alpha=0.3, facecolor="r", edgecolor="r")
                ax.add_patch(polygon)
            fig.canvas.draw()

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    return path


def select_polygon_and_path(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Left click to add points. Right click to undo. Close the window to finish.")

    polygon_path = []
    movement_path = []

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    return polygon_path, movement_path


def select_path(image, polygon, num_frames=49):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.imshow(image)
    ax.set_title("Left click to add points. Right click to undo. Close the window to finish.")

    path = []

    # Add sliders for final scale and rotation
    ax_scale = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_rot = plt.axes([0.25, 0.15, 0.65, 0.03])

    scale_slider = Slider(ax_scale, "Final Scale", 0.1, 5.0, valinit=1)
    rot_slider = Slider(ax_rot, "Final Rotation", -360, 360, valinit=0)

    scales = []
    rotations = []

    def interpolate_transformations(n_points):
        scales = np.linspace(1, scale_slider.val, n_points)
        rotations = np.linspace(0, rot_slider.val, n_points)
        return scales, rotations

    def update_display():
        ax.clear()
        ax.imshow(image)
        ax.set_title("Left click to add points. Right click to undo. Close the window to finish.")

        n_points = len(path)
        if n_points < 1:
            fig.canvas.draw_idle()
            return

        # Interpolate scales and rotations over the total number of points
        scales[:], rotations[:] = interpolate_transformations(n_points)

        origin = np.array(path[0])

        for i in range(n_points):
            ax.plot(path[i][0], path[i][1], "bo")
            if i > 0:
                ax.plot([path[i - 1][0], path[i][0]], [path[i - 1][1], path[i][1]], "b-")
            # Apply transformation to the polygon
            transformed_polygon = apply_transformation(np.array(polygon), scales[i], rotations[i], origin)
            # Offset polygon to the current point relative to the first point
            position_offset = np.array(path[i]) - origin
            transformed_polygon += position_offset
            mpl_poly = Polygon(
                transformed_polygon,
                closed=True,
                alpha=0.3,
                facecolor="r",
                edgecolor="r",
            )
            ax.add_patch(mpl_poly)

        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # Left click
            path.append((event.xdata, event.ydata))
            update_display()
        elif event.button == 3 and path:  # Right click
            path.pop()
            update_display()

    def on_slider_change(val):
        update_display()

    scale_slider.on_changed(on_slider_change)
    rot_slider.on_changed(on_slider_change)

    scales, rotations = [], []

    cid_click = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid_click)

    # Final interpolation after the window is closed
    n_points = num_frames
    if n_points > 0:
        scales, rotations = interpolate_transformations(n_points)
        rotations = [-x for x in rotations]
        path = as_numpy_array(path)
        path = as_numpy_array([linterp(path, i) for i in np.linspace(0, len(path) - 1, num=n_points)])

    return path, scales, rotations


def animate_polygon(image, polygon, path, scales, rotations,interp=cv2.INTER_LINEAR):
    frames = []
    transformed_polygons = []
    origin = np.array(path[0])

    h, w = image.shape[:2]

    for i in eta(range(len(path)), title="Creating frames for this layer..."):
        # Compute the affine transformation matrix
        theta = np.deg2rad(rotations[i])
        scale = scales[i]

        a11 = scale * np.cos(theta)
        a12 = -scale * np.sin(theta)
        a21 = scale * np.sin(theta)
        a22 = scale * np.cos(theta)

        # Compute translation components
        tx = path[i][0] - (a11 * origin[0] + a12 * origin[1])
        ty = path[i][1] - (a21 * origin[0] + a22 * origin[1])

        M = np.array([[a11, a12, tx], [a21, a22, ty]])

        # Apply the affine transformation to the image
        warped_image = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=interp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Transform the polygon points
        polygon_np = np.array(polygon)
        ones = np.ones(shape=(len(polygon_np), 1))
        points_ones = np.hstack([polygon_np, ones])
        transformed_polygon = M.dot(points_ones.T).T
        transformed_polygons.append(transformed_polygon)

        # Create a mask for the transformed polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(transformed_polygon)], 255)

        # Extract the polygon area from the warped image
        rgba_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2BGRA)
        alpha_channel = np.zeros((h, w), dtype=np.uint8)
        alpha_channel[mask == 255] = 255
        rgba_image[:, :, 3] = alpha_channel

        # Set areas outside the polygon to transparent
        rgba_image[mask == 0] = (0, 0, 0, 0)

        frames.append(rgba_image)

    return gather_vars("frames transformed_polygons")


def apply_transformation(polygon, scale, rotation, origin):
    # Translate polygon to origin
    translated_polygon = polygon - origin
    # Apply scaling
    scaled_polygon = translated_polygon * scale
    # Apply rotation
    theta = np.deg2rad(rotation)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_polygon = np.dot(scaled_polygon, rotation_matrix)
    # Translate back
    final_polygon = rotated_polygon + origin
    return final_polygon


def cogvlm_caption_video(video_path, prompt="Please describe this video in detail."):
    import rp.web_evaluator as wev

    client = wev.Client("100.113.27.133")
    result = client.evaluate("run_captioner(x,prompt=prompt)", x=video_path, prompt=prompt)
    if result.errored:
        raise result.error
    return result.value


if __name__ == "__main__":
    fansi_print(big_ascii_text("Go With The Flow!"), "yellow green", "bold")

    image_path = input_conditional(
        fansi("First Frame: Enter Image Path or URL", "blue cyan", "italic bold underlined"),
        lambda x: is_a_file(x.strip()) or is_valid_url(x.strip()),
    ).strip()

    print("Using path: " + fansi_highlight_path(image_path))
    if is_video_file(image_path):
        fansi_print('Video path was given. Using first frame as image.')
        image=load_video(image_path,length=1)[0]
    else:
        image = load_image(image_path, use_cache=True)
        image = resize_image_to_fit(image, height=1440, allow_growth=False)

    prompt=input(fansi('Input the video caption >>> ','blue cyan','bold'))

    SCALE_FACTOR=1
    #Adjust resolution to 720x480: resize then center-crop
    HEIGHT=480*SCALE_FACTOR
    WIDTH=720*SCALE_FACTOR
    image = resize_image_to_hold(image,height=HEIGHT,width=WIDTH) 
    image = crop_image(image, height=HEIGHT,width=WIDTH, origin='center')
    title = input_default(
        fansi("Enter a title: ", "blue cyan", "italic bold underlined"),
        get_file_name(
            image_path,
            include_file_extension=False,
        ),
    )
    output_folder=make_directory(get_unique_copy_path(title))
    print("Output folder: " + fansi_highlight_path(image_path))

    fansi_print("How many layers?", "blue cyan", "italic bold underlined"),
    num_layers = input_integer(
        minimum=1,
    )

    layer_videos = []
    layer_polygons = []
    layer_first_frame_masks = []
    layer_noises = []

    for layer_num in range(num_layers):
        layer_noise=np.random.randn(HEIGHT,WIDTH,18).astype(np.float32)

        fansi_print(f'You are currently working on layer #{layer_num+1} of {num_layers}','yellow orange','bold')
        if True or not "polygon" in vars() or input_yes_no("New Polygon?"):
            polygon = select_polygon(image)
        if True or not "animation" in vars() or input_yes_no("New Animation?"):
            animation = select_path(image, polygon)

        
        animation_output = animate_polygon(image, polygon, *animation)

        noise_output_1 = as_numpy_array(animate_polygon(layer_noise[:,:,3*0:3*1], polygon, *animation, interp=cv2.INTER_NEAREST).frames)
        noise_output_2 = as_numpy_array(animate_polygon(layer_noise[:,:,3*1:3*2], polygon, *animation, interp=cv2.INTER_NEAREST).frames)
        noise_output_3 = as_numpy_array(animate_polygon(layer_noise[:,:,3*2:3*3], polygon, *animation, interp=cv2.INTER_NEAREST).frames)
        noise_output_4 = as_numpy_array(animate_polygon(layer_noise[:,:,3*3:3*4], polygon, *animation, interp=cv2.INTER_NEAREST).frames)
        noise_output_5 = as_numpy_array(animate_polygon(layer_noise[:,:,3*4:3*5], polygon, *animation, interp=cv2.INTER_NEAREST).frames)
        noise_output_6 = as_numpy_array(animate_polygon(layer_noise[:,:,3*5:3*6], polygon, *animation, interp=cv2.INTER_NEAREST).frames)
        noise_warp_output = np.concatenate(
            [
                noise_output_1[:,:,:,:3],
                noise_output_2[:,:,:,:3],
                noise_output_3[:,:,:,:3],
                noise_output_4[:,:,:,:3],
                noise_output_5[:,:,:,:3],
                noise_output_6[:,:,:,:1],
            ],
            axis=3,#THWC
        )

        frames, transformed_polygons = destructure(animation_output)

        mask = get_image_alpha(frames[0]) > 0
        
        layer_polygons.append(transformed_polygons)
        layer_first_frame_masks.append(mask)
        layer_videos.append(frames)
        layer_noises.append(noise_warp_output)

    if True or input_yes_no("Inpaint background?"):
        total_mask = sum(layer_first_frame_masks).astype(bool)
        background = cv_inpaint_image(image, mask=total_mask)
    else:
        background = "https://t3.ftcdn.net/jpg/02/76/96/64/360_F_276966430_HsEI96qrQyeO4wkcnXtGZOm0Qu4TKCgR.jpg"
        background = load_image(background, use_cache=True)
        background = cv_resize_image(background, get_image_dimensions(image))
        background=as_rgba_image(background)

    ###
    output_frames = [
        overlay_images(
            background,
            *frame_layers,
        )
        for frame_layers in eta(list_transpose(layer_videos),title=fansi("Compositing all frames of the video...",'green','bold'))
    ]
    output_frames=as_numpy_array(output_frames)

    
    output_video_file=save_video_mp4(output_frames, output_folder+'/'+title + ".mp4", video_bitrate="max")
    output_mask_file = save_video_mp4(
        [
            sum([get_image_alpha(x) for x in layers])
            for layers in list_transpose(layer_videos)
        ],
        output_folder + "/" + title + "_mask.mp4",
        video_bitrate="max",
    )
    

    ###
    fansi_print("Warping noise...",'yellow green','bold italic')
    output_noises = np.random.randn(1,HEIGHT,WIDTH,16)
    output_noises=np.repeat(output_noises,49,axis=0)
    for layer_num in range(num_layers):
        fansi_print(f'Warping noise for layer #{layer_num+1} of {num_layers}','green','bold')
        for frame in eta(range(49),title='frame number'):
            noise_mask = get_image_alpha(layer_videos[layer_num][frame])[:,:,None]>0
            noise_video_layer = layer_noises[layer_num][frame]
            output_noises[frame]*=(noise_mask==0)
            output_noises[frame]+=noise_video_layer*noise_mask
            #display_image((noise_mask * noise_video_layer)[:,:,:3])
            display_image(output_noises[frame][:,:,:3]/5+.5)
    
    import einops
    import torch
    torch_noises=torch.tensor(output_noises)
    torch_noises=einops.rearrange(torch_noises,'F H W C -> F C H W')        
    #
    small_torch_noises=[]
    for i in eta(range(49),title='Regaussianizing'):
        torch_noises[i]=nw.regaussianize(torch_noises[i])[0]
        small_torch_noise=nw.resize_noise(torch_noises[i],(480//8,720//8))
        small_torch_noises.append(small_torch_noise)
        #display_image(as_numpy_image(small_torch_noise[:3])/5+.5)
        display_image(as_numpy_image(torch_noises[i,:3])/5+.5)
    small_torch_noises=torch.stack(small_torch_noises)#DOWNSAMPLED NOISE FOR CARTRIDGE!

    ###
    cartridge={}
    cartridge['instance_noise']=small_torch_noises.bfloat16()
    cartridge['instance_video']=(as_torch_images(output_frames)*2-1).bfloat16()
    cartridge['instance_prompt']=prompt
    output_cartridge_file=object_to_file(cartridge, output_folder + "/" + title + "_cartridge.pkl")
            
    ###
    
    
    output_polygons_file=output_folder+'/'+'polygons.npy'
    polygons=as_numpy_array(layer_polygons)
    np.save(output_polygons_file,polygons)
    
    print()
    print(fansi('Saved outputs:','green','bold'))
    print(fansi('    - Saved video: ','green','bold'),fansi_highlight_path(get_relative_path(output_video_file)))
    print(fansi('    - Saved masks: ','green','bold'),fansi_highlight_path(get_relative_path(output_mask_file)))
    print(fansi('    - Saved shape: ','green','bold'),fansi_highlight_path(output_polygons_file))
    print(fansi('    - Saved cartridge: ','green','bold'),fansi_highlight_path(output_cartridge_file))

    print("Press CTRL+C to exit")


    display_video(video_with_progress_bar(output_frames), loop=True)
