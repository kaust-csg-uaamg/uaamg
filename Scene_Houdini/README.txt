This folder contains Houdini files used to generate the fluid simulation scenes and to render the generated results.

We recommend rendering the scene with OpenColorIO (OCIO) color configurations.
This step is completely optional.

For windows:
1. Download the "Sample OCIO Configurations" from OCIO website:
https://opencolorio.readthedocs.io/en/latest/quick_start/downloads.html
Or directly from the link:
http://github.com/imageworks/OpenColorIO-Configs/zipball/master

For example, the zip file we got is "[PathToDownloadDirectory]\imageworks-OpenColorIO-Configs-v1.0_r2-8-g0bb079c.zip"
Extract it.

2. Add the OCIO config file path in the Houdini environment file: "houdini.env"
The environment file can be found in Documents, for example:
"C:\Users\[MyUserName]\Documents\houdini19.0\houdini.env"

Add the following file path to "houdini.env" to set the OCIO config file:
OCIO = "[PathToDownloadDirectory]\imageworks-OpenColorIO-Configs-0bb079c\aces_1.0.3\config.ocio"

That's it, OCIO is now supported by Houdini.

When an image is rendered in Houdini, we suggest keep it as .exr file or .pic file to keep the full color information.
However, a look up table (lut) is required to correctly save the picture in .jpg or .png format.
The look up table is "\aces_1.0.3\baked\houdini\sRGB for ACEScg Houdini.lut". Which output an image in SRGB format.

Try the following steps in Houdini to convert images to .jpg format
1. In /obj network panel type "COP2" and create a composite network "cop2net1"
2. Go inside "cop2net1", type "file" to read image sequences as input.
3. Type "ROP" select "ROP File Output". In its parameter panel, uncheck "Convert to image format's colorspace", and choose the Output LUT to the previous look up table.




