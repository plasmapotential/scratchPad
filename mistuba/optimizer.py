#uses gradient descent to reproduce an original image after we perturbed the reference

import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant('llvm_ad_rgb')

scene = mi.load_file('/home/tom/Downloads/scenes/cbox.xml', res=128, integrator='prb')

image_ref = mi.render(scene, spp=1024)

# Preview the reference image
mi.util.convert_to_bitmap(image_ref)

mi.Bitmap(image_ref).write('cbox_ref.exr')

fig = plt.figure(figsize=(1, 3))
fig.add_subplot(1,3,1).imshow(image_ref); plt.axis('off'); plt.title('original')

params = mi.traverse(scene)

key = 'red.reflectance.value'

# Save the original value
param_ref = mi.Color3f(params[key])

# Set another color value and update the scene
params[key] = mi.Color3f(0.01, 0.2, 0.9)
params.update();

image_init = mi.render(scene, spp=128)
mi.util.convert_to_bitmap(image_init)

fig.add_subplot(1,3,2).imshow(image_init); plt.axis('off'); plt.title('blue')



opt = mi.ad.Adam(lr=0.05)
opt[key] = params[key]
params.update(opt);

def mse(image):
    return dr.mean(dr.sqr(image - image_ref))

iteration_count = 50

errors = []
for it in range(iteration_count):
    # Perform a (noisy) differentiable rendering of the scene
    image = mi.render(scene, params, spp=4)

    # Evaluate the objective function from the current rendered image
    loss = mse(image)

    # Backpropagate through the rendering process
    dr.backward(loss)

    # Optimizer: take a gradient descent step
    opt.step()

    # Post-process the optimized parameters to ensure legal color values.
    opt[key] = dr.clamp(opt[key], 0.0, 1.0)

    # Update the scene state to the new optimized values
    params.update(opt)

    # Track the difference between the current color and the true value
    err_ref = dr.sum(dr.sqr(param_ref - params[key]))
    print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}", end='\r')
    errors.append(err_ref)
print('\nOptimization complete.')


image_final = mi.render(scene, spp=128)
fig.add_subplot(1,3,3).imshow(image_final); plt.axis('off'); plt.title('final')

plt.show()

plt.plot(errors)
plt.xlabel('Iteration'); plt.ylabel('MSE(param)'); plt.title('Parameter error plot');
plt.show()