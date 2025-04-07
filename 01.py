#%%

import vtk
import imageio.v3 as iio
import numpy as np
#import cupy as cp
import os
import json



foldername = "Sample_images"

foldername_snapshots = "snapshots/"


n_snapshot = 1








def load_data(foldername):
    print("reading .tiff files ...")
    tiff_files = sorted([f for f in os.listdir(foldername) if f.endswith(".tiff") or f.endswith(".tif")])
    print(tiff_files)
    images = []
    for file in tiff_files:
        image_path = os.path.join(foldername, file)
        img = iio.imread(image_path)
        images.append(img)
    image_stack = np.stack(images, axis=0)
    return image_stack


def normalize(array):
    array_min = np.min(array[:])
    array_max = np.max(array[:])
    array = (array - array_min ) / (array_max - array_min)
    return array


def filter_contrast(numpy_array, kernel_size = 5):
    print("filtering array using contrast-filter ...")
    array_size = (numpy_array.shape[0], numpy_array.shape[1], numpy_array.shape[2])
    #data = cp.random.rand(*array_size, dtype=cp.float32)
    data = cp.asarray(numpy_array, dtype=cp.float32)
    half_kernel = kernel_size // 2
    padded_data = cp.pad(data, pad_width=half_kernel, mode='constant', constant_values=0)
    output_data = cp.empty_like(data)
    for z in range(half_kernel, array_size[0] + half_kernel):
        for y in range(half_kernel, array_size[1] + half_kernel):
            for x in range(half_kernel, array_size[2] + half_kernel):
                region = padded_data[z-half_kernel:z+half_kernel+1, y-half_kernel:y+half_kernel+1, x-half_kernel:x+half_kernel+1] # extract kernel
                region_min = cp.min(region)
                region_max = cp.max(region)
                center_value = padded_data[z, y, x]
                output_data[z-half_kernel, y-half_kernel, x-half_kernel] = (center_value - region_min) / (region_max - region_min)
    output_numpy_array = cp.asnumpy(output_data)
    return output_numpy_array







    
def numpy_to_vtk_image(data):
    importer = vtk.vtkImageImport()
    data_string = data.tobytes()
    importer.CopyImportVoidPointer(data_string, len(data_string))
    importer.SetDataScalarTypeToUnsignedChar()
    importer.SetNumberOfScalarComponents(1)
    importer.SetDataExtent(0, data.shape[2]-1, 0, data.shape[1]-1, 0, data.shape[0]-1)
    importer.SetWholeExtent(0, data.shape[2]-1, 0, data.shape[1]-1, 0, data.shape[0]-1)
    return importer



def create_volume_rendering_ultrasound(vtk_image):
    # Transfer function (mapping scalar values to opacity and color)
    opacity_transfer_function = vtk.vtkPiecewiseFunction()
    opacity_transfer_function.AddPoint(0, 0.0)
    opacity_transfer_function.AddPoint(1, 0.0)
    opacity_transfer_function.AddPoint(10, 0.1)
    opacity_transfer_function.AddPoint(100, 0.2)
    opacity_transfer_function.AddPoint(255, 0.2)
    #opacity_transfer_function.AddPoint(100, 0.0)
    #opacity_transfer_function.AddPoint(150, 0.1)
    #opacity_transfer_function.AddPoint(255, 0.2)

    color_transfer_function = vtk.vtkColorTransferFunction()
    color_transfer_function.AddRGBPoint(0.0, 0.0, 0.0, 0.0)  
    color_transfer_function.AddRGBPoint(10, 0.5, 0.5, 0.5)  
    color_transfer_function.AddRGBPoint(50, 1.0, 1.0, 1.0)  
    color_transfer_function.AddRGBPoint(255, 1.0, 1.0, 1.0)  
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_transfer_function)
    volume_property.SetScalarOpacity(opacity_transfer_function)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper.SetInputConnection(vtk_image.GetOutputPort())
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    return volume


















# Create a renderer, render window, and interactor
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)








image = load_data(foldername)
image = np.float32(image)
image = normalize(image)
#image = filter_contrast(image, 5)
image = image * 255
image = np.uint8(image)

nx = image.shape[0]
ny = image.shape[0]
nz = image.shape[0]

vtk_image = numpy_to_vtk_image(image)
volume = create_volume_rendering_ultrasound(vtk_image)


renderer.AddVolume(volume)



camera = vtk.vtkCamera()
camera.SetPosition(2000, ny/2, nz/2)  # Set the camera position (x, y, z)
camera.SetFocalPoint(nx/2, ny/2, nz/2)  # Set the focal point to the origin
camera.SetViewUp(0, 1, 0)  # Set the view-up vector (typically along the Y-axis)

# Attach the camera to the renderer
renderer.SetActiveCamera(camera)




























def save_screenshot(render_window):
    global n_snapshot
    filename = foldername_snapshots + 'screenshot_' + f'{n_snapshot:04}' + '.png'
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.SetScale(1)  # Image scaling
    window_to_image_filter.SetInputBufferTypeToRGB()  # Captures the alpha (RGBA)
    window_to_image_filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()
    n_snapshot = n_snapshot + 1

















def handle_keypress(obj, event):
    key = interactor.GetKeySym()  
    shift_pressed = interactor.GetShiftKey()
    if key == "v":
        current_visibility = volume.GetVisibility()
        volume.SetVisibility(not current_visibility)        
    elif key == "r":
        print("hello")
    elif key == "t" and not shift_pressed:
        print("hello")
    elif key == "w":
        save_screenshot(render_window)
    render_window.Render() 
















interactor.AddObserver("KeyPressEvent", handle_keypress)

# Set up the render window
renderer.SetBackground(0.0, 0.0, 0.0)  # Background color

camera_light = vtk.vtkLight()
camera_light.SetLightTypeToSceneLight()  # Make the light follow the camera
camera_light.SetColor(1.0, 1.0, 1.0)      # Set the light color (white)
camera_light.SetIntensity(0.9)   

light_top = vtk.vtkLight()
light_top.SetLightTypeToCameraLight()  # Make the light follow the camera
light_top.SetColor(1.0, 1.0, 1.0)      # Set the light color (white)
light_top.SetIntensity(0.9) 
light_top.SetPosition(-500,ny/2,nz/2)  

renderer.AddLight(camera_light)
renderer.AddLight(light_top)




render_window.SetSize(1000, 1000)

# Start the interaction
render_window.Render()
interactor.Initialize()
interactor.Start()



# %%
