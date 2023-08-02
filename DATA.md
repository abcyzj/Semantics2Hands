# Data Preparation

## Annotate *Twist-Bend-Splay* Frames for MANO

Run the following commands:
```bash
mkdir -p artifact/MixHand/finger_data
python -m run.calculate_mano_axis --output artifact/MixHand/finger_data/mano_finger_axis.pkl
```
The annotation script would display the current `Splay Axis` (Red) and `Bend Axis` (Black). Press `,` and `.` to adjust the axis orientation. Press `Q` to confirm the axis orientation of the current finger. When you are done annotating all 10 fingers, the orientation data will be saved in `artifact/MixHand/finger_data/mano_finger_axis.pkl`.

## Donload InterHand2.6M data

- Download annotations from [InterHand2.6M](https://drive.google.com/drive/folders/12RNG9slv9i_TsXSoZ6pQAq-Fa98eGLoy) and extract it into `artifact/InterHand2.6M`. Run:
```bash
python -m run.convert_interhand_bvh --input artifact/InterHand2.6M/annotations/train/InterHand2.6M_train_MANO_NeuralAnnot.json --output_dir artifact/InterHand2.6M/annotations/train/bvh
```

- Categorize bvh files in `artifact/InterHand2.6M/annotations/train/bvh` by subject and place them in `Mano0`~`Mano9` subdirectories in `artifact/Mixhand`.


## Download data from the Mixamo website

- Download Mixamo fbx files following instructions in [SAN](https://github.com/DeepMotionEditing/deep-motion-editing). Make a subdirectory in `artifact/MixHand` or each character and download fbx files (withou skin) into these subdirectories. Run `blender -b -P blender_scripts/fbx2bvh.py -- --input_fbx in.fbx --output_bvh out.bvh` to convert each fbx file to a bvh file.
- Download the rest-pose fbx file (with skin) for each character and place it in `artifact/MixHand/finger_data`. Then for each character, run `blender -b -P blender_scripts/fbx2mesh.py -- --input ./artifact/MixHand/finger_data/Kaya.fbx --output artifact/MixHand/finger_data/Kaya_mesh_data.npz`, where `Kaya` is the name of the character.


## Annotate *Twist-Bend-Splay* Frames for Mixamo

For each Mixamo character, run:
```bash
python -m run.calculate_mixamo_axis --mesh_data artifact/MixHand/finger_data/Kaya_mesh_data.npz --output artifact/MixHand/finger_data/Kaya_finger_axis.pkl
```
For different characters, replace the name `Kaya` with the name of the corresponding character. The annotation script would display the current `Splay Axis` (Red) and `Bend Axis` (Black). Press `,` and `.` to adjust the axis orientation. Press `Q` to confirm the axis orientation of the current finger. When you are done annotating all 10 fingers, the orientation data will be saved in `artifact/MixHand/finger_data/Kaya_finger_axis.pkl`.

## Preprocess the whole dataset

Run `python -m run.preprocess_mixhand`.
