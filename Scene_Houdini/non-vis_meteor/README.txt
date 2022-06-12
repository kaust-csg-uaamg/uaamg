1. generate the meteor geometry, static container, and the initial liquid SDF in Houdini using "meteor_generate_scene.hipnc"
meteor_dynamic contains the meteor geometry for each frame.
meteor_dynamic_from_paper contains the meteor geometry that we used in the paper, but unfortunately we cannot generate the exact meteor motion from Houdini rigid body sim again.

2. run simulation with "config_meteor_lowres.yaml" 
..\..\bin\FLIP_simulator\FLIP_simulator.exe config_meteor_lowres.yaml

or "config_meteor_paper.yaml"
..\..\bin\FLIP_simulator\FLIP_simulator.exe config_meteor_paper.yaml

The simulation results will be in ".\output"

3. render the simulation result in Houdini with "meteor_scene_render.hipnc"

