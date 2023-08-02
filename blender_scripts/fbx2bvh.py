import sys
from argparse import ArgumentParser

import bpy


def delete_hierarchy(obj):
    names = set([obj.name])

    # recursion
    def get_child_names(obj):
        for child in obj.children:
            names.add(child.name)
            if child.children:
                get_child_names(child)

    get_child_names(obj)

    objects = bpy.data.objects
    for n in names:
        objects[n].select_set(True)

    bpy.ops.object.delete()


def main(args):
    bpy.ops.import_scene.fbx(filepath=args.input_fbx, automatic_bone_orientation=True)
    a = bpy.context.object.animation_data.action
    frame_start, frame_end = int(a.frame_range[0]), int(a.frame_range[1])
    bpy.ops.export_anim.bvh(filepath=args.output_bvh, frame_start=frame_start, frame_end=frame_end, root_transform_only=True)
    delete_hierarchy(bpy.data.objects['Armature'])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_fbx', type=str, required=True)
    parser.add_argument('--output_bvh', type=str, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    main(args)
