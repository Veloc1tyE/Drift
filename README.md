# Drift
Deep Reinforcement Learning for Fusion Control. This is an implementation of the World-Models approach to control, and is built on top of the BOUT++ 
open source fusion simulation library.

The aim is to demonstrate the ability of reinforcement learning algorithms to predict and control disruptions in fusion reactors.

The feedback mechanism is the expected return, which is just the energy output minus the input over the course of an episode.

The control mechanism is Electron Cyclotron Resonance Heating (ECRH), from which the agent aims to prevent and control instabilities,
while maximising energy output over time.


