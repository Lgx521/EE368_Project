# Lab project 3 - Jacobian


## Plot for the cartesian velocity
The video illustrate the tool(or end effector)'s cartesian velocity `v_x`, `v_y` and `v_z`.  
As we can see in the video, as the e.e. moves, the speed varies as well.

## Kinematic Verification
The code for calculating the forward kinematics is totally correct.   
DH parameter processing: The code correctly implements the standard DH parameter method, and the transformation matrix calculation of each link conforms to DH conventions.  
Transformation matrix combination: The total transformation matrix of the end effector is obtained by continuously multiplying the transformation matrices of each connecting rod, which is the standard method of forward kinematics.  
Euler angle calculation: The code implements the calculation of $ZYZ$ Euler angle and handles angle range limitations to ensure that the results are within a reasonable range.  
Gen3 Lite robotic arm parameters: The provided DH parameters are consistent with the actual parameters of the Kinova Gen3 Lite robotic arm, including the length of each link and joint offset.  
Output verification: The code outputs the position $(x, y, z)$ and pose $(α, β, γ)$ of the end effector, which is a complete 6-degree-of-freedom pose description.  