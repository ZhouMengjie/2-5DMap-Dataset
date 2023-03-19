import bpy
import sys
import os

# test on blender 3.1.0
# usage: blender --background --python blender_osm.py -- xxx (file name)

argv = sys.argv[sys.argv.index("--") + 1:]
assert(len(argv) == 1)
data_path = os.path.join(os.getcwd(), 'datasets', argv[0])
osm_file = os.path.join(data_path, (argv[0] + '.osm'))
assert os.path.exists(osm_file), 'Cannot open .osm file: {}'.format(osm_file)
txt_file = os.path.join(data_path, (argv[0] + '.txt'))
if not os.path.isdir(os.path.join(data_path, (argv[0]+'_obj'))):
    os.makedirs(os.path.join(data_path, (argv[0]+'_obj')))

while bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects[0], do_unlink=True)

bpy.context.scene.blosm.dataType = 'osm'
bpy.context.scene.blosm.osmSource = 'file'
bpy.context.scene.blosm.osmFilepath = osm_file

bpy.context.scene.blosm.mode = '3Dsimple'

bpy.context.scene.blosm.buildings = True
bpy.context.scene.blosm.water = True
bpy.context.scene.blosm.forests = True
bpy.context.scene.blosm.vegetation = True
bpy.context.scene.blosm.highways = True
bpy.context.scene.blosm.railways = True

bpy.context.scene.blosm.defaultRoofShape = 'flat'
bpy.context.scene.blosm.levelHeight = 3.0
bpy.context.scene.blosm.singleObject = True

bpy.context.scene.blosm.defaultLevels[0].weight = 10
bpy.context.scene.blosm.defaultLevels[1].weight = 20
bpy.context.scene.blosm.defaultLevels[2].weight = 30

bpy.ops.blosm.import_data()

# check the geographical coordinates:
scene = bpy.context.scene
print(['global origin at:' + str(scene["lat"]) + ',' + str(scene["lon"])])
f = open(txt_file, 'w')
f.write(str(scene["lat"])+'\n')
f.write(str(scene["lon"])+'\n')
f.close()

bpy.ops.object.select_all(action='SELECT')
obs = bpy.context.selected_objects
for ob in obs:
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(state=True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.convert(target='MESH')

    print(ob.name)
    output_file = os.path.join(data_path, argv[0]+'_obj', (ob.name + '.obj'))
    bpy.context.view_layer.objects.active = ob
    try:
        bpy.ops.object.modifier_add(type='TRIANGULATE')
        bpy.context.object.modifiers["Triangulate"].quad_method = 'SHORTEST_DIAGONAL'
        bpy.ops.object.modifier_apply(modifier="Triangulate")
        bpy.ops.export_scene.obj(filepath=output_file, use_selection=True, use_materials=False)
    except:
        pass
