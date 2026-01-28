# PPO-Proxy-federated-Learning-for-3D-path-planing
Proximity Federated Learning (PFL) is another strategy to utilize PPO agents to investigate the environment and propose the path properly as the outcome. The parameters to find the path using PFL + PPO are shown as follows:

	1. The rollout step is set to 1024.
	2. The minibatch size is set to 1024.
	3. The clip coefficient is set to 0.2.
	4. The learning rate is set to 0.0003.
	5. Number of epochs for each model is set 500.
	6. The maximum number of episodes is set to 2500 for 40 obstacles and 300 obstacles.
	7. The evaluated steps are considered as 5.
	8. Optimizer is Adam optimizer.
	9. The entropy coefficient is set 0.01.
	10. The critic loss coefficient is set to 0.9405.
	11. PFL coefficient is set to 0.01.

  
The proposed solution implements a standard PPO actor–critic loop inside a federated learning scaffold: each client runs its own 3D environment with different sets of obstacles, collects a fixed-size rollout buffer (1024), computes GAE advantages and returns, and performs multiple local PPO epochs (policy surrogate with clipping, value Mean Square Error (MSE), and entropy bonus) on GPU before the server aggregates models via AFL. 
The actor is a CNN–GRU that maps the 6 D observation to a 26 way categorical policy and the critic is a CNN + multi head attention value head; during local updates the code optionally adds a PFL proximal term (\frac{\mu}{2})∥W- W_{global}∥2 to the minibatch loss so each client is penalized for drifting too far from the broadcast global parameters. Concretely, the server first computes and broadcasts a global state, clients collect rollouts and run local PPO update, then the server averages the updated client weights and rebroadcasts the aggregated state for the next round this cycle encourages shared knowledge while still allowing local exploration and adaptation.
