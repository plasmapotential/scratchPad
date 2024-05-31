import unreal
import csv

# Path to the CSV file
file_path = "/home/tlooby/projects/UE5/Prad_all.csv"

# Scale factor for MW to light intensity conversion
scale_factor = 10.0

# Function to create a point light in Unreal Engine
def create_point_light(location, intensity):
    # Create a new point light actor
    point_light = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PointLight, location)
    # Set the light intensity
    point_light.set_editor_property('intensity', intensity)
    return point_light

# Open the CSV file and read the data
with open(file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    # Skip header row if there is one
    next(csvreader)
    for row in csvreader:
        x, y, z, mw = map(float, row)
        location = unreal.Vector(x, y, z)
        intensity = mw * scale_factor
        create_point_light(location, intensity)
