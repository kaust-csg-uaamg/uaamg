1. Use Houdini and meteor_generate_scene.hipnc to generate the simulation geometry
Must use "generate_meteor_motion" geometry node to generate the moving meteor object.

2. run simulation with 
..\..\bin\FLIP_simulator\FLIP_simulator.exe config_meteor_low_res.yaml

or 
..\..\bin\FLIP_simulator\FLIP_simulator.exe config_meteor_paper_highres.yaml


3. Use vis_meteor_scene_render.hipnc to render the result.