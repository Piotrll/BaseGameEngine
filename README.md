This repository Contains Basic foundation of a 3D Game Engine based on OpenGL and pygame
For now without API
While using it you can move freely through 3D space and load objects with .obj extension
They should be contained in folder "objects" and appended in file w/o extension as text in format below
nameOfObject.obj 1 2 1 1 1

1. Name of the file
2. X Y Z Coordinates
3. Fifth one represents if Object is static or not, thou as I'am writing this I have not yet implemented
   moving objects beside camera control, so 1 is static and 0 is dynamic (future me concern)
4. Texture id corresponding to the one in texList
