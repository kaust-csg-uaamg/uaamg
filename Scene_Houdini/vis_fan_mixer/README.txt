1. Use Houdini and fan_mixer_scene_generate.hipnc to generate the scene
Must go into the "rotating_turbine_fan_blade" geometry node to generate the rotating fan geometry

2. run simulation with 
..\..\bin\FLIP_simulator\FLIP_simulator.exe config_fan_mixer.yaml

or 
..\..\bin\FLIP_simulator\FLIP_simulator.exe config_fan_mixer_low_res.yaml

2. Use fan_mixer_viscous_render.hipnc to render the scene