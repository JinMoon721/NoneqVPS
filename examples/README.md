1. generateDrivenEnsemble.py
using the most recent model in ../models/, generate doob-force driven ensemble of
trajectories. The trajectory length should correspond to the model's total time range.
Initial configurations for trajectory is extracted from ../data/equilibriumSamples.
If the directory is empty, run long-enough equilibrium trajectories and save equilibrium
snapshots to the directory.


2. visualizeDoob.py
read the most recent model, plot time-dependent doob-potential, 
-kBTlnq(r, t)


3. initialTPS.py
generate initial TPS trajectory by brute force equilibrium simulation
the initial brute-force reactive trajectory will be saved in ./data/TPS/original.traj
Later, TPS will be performed on top of this original trajectory
Caution : this example only produce reactive part of the trajectory: starting from the
particle touched outer boundary until it touches inner boundary.
>>> ../data/original.traj


4. runTPS.py / runTPS.sh
generate TPS reactive trajectory ensemble, >> save train.npz, metadata for trajectory
re-generation, and initial snapshot samples for next TPS run
parallized version : runTPS.sh


5. merge_trainset.py
merge trainsets in ../data/TPS/worker_{i}/* into batched positions and noises
>> ../data/trainSet

6. trainModel.py
training model, starting from last ckpt file saved in ../data/models/

7. trainBiasedModel.py
Using 100 biased trajectories and random 100 TPS trajectories,
retrain the model with the real variational bound
