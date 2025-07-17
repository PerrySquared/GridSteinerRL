# GridSteinerRL

## General Information
- The goal of this project is to create an RL (currently A2C) approach to solve the global routing problem (uses rectilinear approach with layer selection to create paths between pins)
- Support up to 5 pins (could be increased with additional model training)
- Requires matrix format for input (.lef + .def -> matrix)

<p>A quick example of a result. Red - created path, Other - usage heatmap of 3D matrix cells, i.e. how many paths were routed through them.</p>
<img width="600" height="440" alt="image" src="https://github.com/user-attachments/assets/925462f7-be09-4e31-a4e4-04293897b137" />
