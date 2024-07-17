import mitsuba as mi
import numpy as np
import matplotlib as plt


# Initialize Mitsuba
mi.set_variant('llvm_ad_rgb')  # or 'cuda_ad_rgb' for GPU rendering



def create_scene(emission_points, cad_mesh_path):
    # Create the scene dictionary
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 10
        },
        'sensor': {
            'type': 'perspective',
            'to_world': mi.ScalarTransform4f.look_at(origin=[0, 5, 10], target=[0, 0, 0], up=[0, 1, 0]),
            'film': {
                'type': 'hdrfilm',
                'width': 1280,
                'height': 720
            }
        }
    }

    # Add the CAD mesh
    scene_dict['mesh'] = {
        'type': 'ply',
        'filename': cad_mesh_path,
    }

    # Add emission sources
    for i, point in enumerate(emission_points):
        scene_dict['emitter_{:03d}'.format(i)] = {
            'type': 'point',
            'position': point,
            'intensity': {
                'type': 'spectrum',
                'value': 1.0,
                }
        }



    return mi.load_dict(scene_dict)

# Define emission points and CAD mesh path
emission_points = [(1720.0, -460.0, -1470.0), (1720.0, -450.0, -1470.0)]

objMesh = '/home/tlooby/HEATruns/SPARC/xTarget_jan2024/CAD/T5Cmesh_v4.ply'
# Create the scene
scene = create_scene(emission_points, objMesh)

# Render the scene
image = mi.render(scene)

# Save the rendered image
mi.Bitmap(image).write("rendered_image.exr")

fig = plt.figure(figsize=(1, 1))
fig.add_subplot(1,1,1).imshow(image); plt.axis('off'); plt.title('original')
plt.show()