# RAFAEL Machine Learning Challenge 2019 winning solution

This algorithm won the [**RAFAEL Machine Learning Challenge 2019**](http://portal.rafael.co.il/MLchallenge2019/Documents/index.html) competition.

Table of contents:

* [The Competition](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#the-competition)
* [The Solution](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#the-solution) 
  * [Phase 1 - Neural Networks Training](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#phase-1---neural-networks-training)
  * [Phase 2 - Solution Algorithm](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#phase-2---solution-algorithm)
    * [Hyperparameter Tuning](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#hyperparameter-tuning)
* [Algorithm Performance](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#algorithm-performance)
* [Dependencies](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#dependencies) 

## The Competition

[RAFAEL Advanced Defense Systems Ltd.](https://www.rafael.co.il/) 
is a well-known Israeli defense tech company ([wiki](https://en.wikipedia.org/wiki/Rafael_Advanced_Defense_Systems)).

RAFAEL constructed a [game](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution/blob/master/Interceptor_V2.py) 
in which you play a defending turret, which intercepts enemy rockets fired at two cities.  

The [RAFAEL Machine Learning Challenge 2019](http://portal.rafael.co.il/MLchallenge2019/Documents/index.html) 
competition was to write an ML-based python software that plays the game independently, and gets the highest score.

## The Solution

**Approach**: breaking down the challenge into small tasks involving a 
single rocket \ single rocket-interceptor pair. 

### Phase 1 - Neural Networks Training

Neural Networks:
* (1) **Fire-angle assessment network** - for a single rocket.
* (2) **Collision assessment networks** - for a single rocket.
  * binary classification (ground \ city)
  * time-steps to collision
* (4) **Interception assessment networks** - for a single rocket (pre-fire interception assessment) or for a single rocket-interceptor pair (interception assessment). 
  * binary classification (miss \ hit)
  * time-steps to interception (only in case of a successful interception)

### Phase 2 - Solution Algorithm

A decision-making algorithm, which chooses the highest-profile rocket (target) according to:
* **Interceptability**.
* **Threat level** - hitting city\ground, time-steps to collision.
* **Relevancy** - whether the missile is already on its way to be intercepted 
(when an interceptor was already fired).
* **Opportunity level** - proximity to fire angle, ability to fire 
(when not in cooldown mode).

one of the main challenges here was achieving a correct interpretation 
of the rockets & interceptors lists in the observation vector 
(and specifically identifying appearance & disappearance of rockets & interceptors).

#### Hyperparameter Tuning

##### Denominator testing (with initial angle 6°, np seed: 28)

* config 0 - `denominator *= (t_to_fire_angle + 1)`. 
Handles the `t_to_fire_angle == 0` problem.
* config 1 - `denominator *= (t_to_fire_angle + 1) if t_to_fire_angle > fire_action_range else 1`. 
Further penalizes high `t_to_fire_angle` missiles.
* config 2 - `denominator *= (t_to_fire_angle + 1) if t_to_fire_angle >= fire_action_range + fire_threshold else 1`. 
Further penalizes only extremely high `t_to_fire_angle` missiles.

The Denominator was config 1 -

<p float="left">
  <img src="https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution/blob/master/phase_2_solution_algorithm/results/denominator_config.png" width="650">
</p>

##### Initial angle optimization (with Denominator config 1)

The initial angle is the desired barrel angle, when there are no enemy rockets in the air.

The optimal initial angle was 60° -

. | avg | median
--- | --- | ---
**6°** | 9.876 | 12.0
**48°** | 5.926 | 26.5
**54°** | 16.173 | 15.5
**60°** | 32.483 | 69.0
**66°** | 24.423 | 51.0
**72°** | 17.506 | 39.0
**78°** | 22.546 | 28.5
**84°** | 19.613 | 40.0

<p float="left">
  <img src="https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution/blob/master/phase_2_solution_algorithm/results/init_ang.png" width="550">
</p>

## Algorithm Performance

The following video presents the performance of the **first version** of the algorithm 
(a video of the final version of the algorithm will be uploaded shortly):

<p float="left">
  <img src="https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution/blob/master/phase_2_solution_algorithm/results/algorithm_performance.gif" width="500">
</p>

## Dependencies
* Python 3.6.5
* Tensorflow 1.10.0
* Keras 2.2.2
* Numpy
* Matplotlib
