# Testing the SRResCycGAN:
- Platform required to run the test code: 
	- Pytorch [1.3.1] 
	- Python [3.7.3] 
	- Numpy [1.16.4] 
	- cv2 [4.1.1] 

- Testing:
	- Run file named "test_srrescycgan.py" to produce SR results for the AIM2020 Real-Image SR challange (x4).
- Contained Directories information: 
	- models: SRResCycGAN Network structure.
	- modules: Supporting functions for the model.
	- trained_nets_x4: SRResCycGAN trained network
	- LR: Given LR images .
	- sr_results_x4: produced output images of the network saved here. 
