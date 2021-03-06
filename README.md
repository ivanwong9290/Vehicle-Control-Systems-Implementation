# Vehicle Control Systems Implementation
 Project demonstrating Model Reference Adapative Control (MRAC) taking over as the original Linear Quadratic Regulator (LQR) fails to adapt to the loss of thrust in drone.
 
 ## Trajectory Comparison Plot
 The blue curve represents the trajectory from a non-adaptive controller (LQR), while the adaptive controller (MRAC) is able to rectify the trajectory to match the ground truth shortly after experiencing the loss of thrust.
![results](https://user-images.githubusercontent.com/71652695/127975697-703fd752-07e0-475a-bf2c-38ee775098ca.png)

## Video Demonstration
Refer to demo.mp4 for a video of the drone flight under the influence of an adaptive controller.
