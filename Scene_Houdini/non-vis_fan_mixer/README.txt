1. generate the rotating fan geometry and the initial liquid SDF using "fan_mixer_scene_generate.hipnc"
dynamic_geo_f5 contains the fan geometry for each frame.

2. run simulation with "config_fan_mixer_lowres_36M_particle.yaml"
..\..\bin\FLIP_simulator\FLIP_simulator.exe config_fan_mixer_lowres_36M_particle.yaml

or "config_fan_mixer_paper_onebillion.yaml"
..\..\bin\FLIP_simulator\FLIP_simulator.exe config_fan_mixer_paper_onebillion.yaml

The simulation results will be in ".\output_37million" or ".\output_onebillion" depending on configurations.

3. render the simulation result in Houdini with "fan_mixer_render.hipnc"

