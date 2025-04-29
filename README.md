# C2D2.ViewBall
Dense complex number propagating and branchless FFT-based convolutional neural network in Zig utilizing dynamic padding

Yo this is cool. Did I mention it's branchless?
(The network is, but there is one base case in the image processing class's recursion with an 'if' statement).

Example creation (Used to classify some cx28x28 images):

![Creation Example](images/SampleUsage.png)
![Init Example](images/SampleInit.png)

Enjoy!

