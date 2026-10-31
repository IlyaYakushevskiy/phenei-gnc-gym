# phenei-gnc-gym
learning policy for take off and landing of moon hopper

Text to LLM : 
alright, the goal of this project is following. 
We want to simulate control of the landing to the moon of our hopper in the setup of the open ai kinda style gym.  The simulation will help to answer qestions about the thruster and sensors placement, simulate the noise we can get from sensors etc. Secondary purpose if for me (Ilya) to learn about the RL methods used in GNC of the spacecraft.  The deliverable of this project will be a video which has telemetry of the sensor, trained policy and hopper executing landing maneuver. We should combine it with some MC style environment, where sensors have errors generated within specification, as well as randomised parametrisation of landing surface (within reasonable) 

 We will start with simplified model of our hopper. I loaded the cad version of primary structure but the scale is wrong so far. for this sim we can use : 
- 4 main thruster engines 4 sides sides. 
- 600mm payload to ground 
- height total 1.2m 
- 1.7m Diameter upper deck, lower deck diameter 2.0m 
- probably cold gas thruster on the top of the structure, we didn't plan this yet but we take now the Firefly Blue Ghost as main example here

from the /home/ilya/phenei-gnc-gym/design_docs/PheNei-report-MCR-local_copy.pdf you can find section about sensors we planed to use,
stereo cameras and lidar, sensor reading should be simulated

from the chapter about mission analysis you can see at which delta V and angle , orbit eccentricity we approach the landing. Simulation will not simulate ballistic hop but only starting from the approach phase. 

Very Important part of this project is keep all the assumption documented in the most simple form , just like csv document. E.g. , diameter of nozzles are up to you but they should be documented in list "parameters" /home/ilya/phenei-gnc-gym/design_docs/params.csv , also mark how solid is the assumption (made up by llm or take from requirement/document) You must also document trades in .csv files. If you do a decision which is a close call, make a quick trade table between 2 sensors architertures etc 

I dont want to set in stone the framework we use yet, and I want to hear your reocmendation regarding it. We have 2080 Super Nvidia GPU which we can utilise even for 12hrs in automatic mode. What is the best libraries to initiate our gym. I'd also prefer some rather funcitional but beautiful animation, should we use sth like openCV, or tell me what is standiard in industry, I want to also learn current best robotics approches. I'm well familiar with python and pytorch tho , I think it can be a good starting point, but I also want to learn ETH RSL and Aerospace way. 


Open questions: 
- 2D or 3D ? Adding 1 dimension does not increase relative complexity by far but is more realistic in terms of DOF 
- framework and software to use? 
- planing, MC in the end, what should be the starting point? 
- what parameters should be fixed in stone to start the project?e.g.  What is theortical amount of thruster we need from classical control perspective? 