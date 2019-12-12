# RAFAEL Machine Learning Challenge 2019 winning solution

This solution won the [**RAFAEL Machine Learning Challenge 2019**](http://portal.rafael.co.il/MLchallenge2019/Documents/index.html) competition.

Table of contents:

* [The Competition](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#the-competition)
* [The Solution](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#the-solution) 
  * [Phase 1 - Neural Networks Training](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#phase-1---neural-networks-training)
  * [Phase 2 - Solution Algorithm](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#phase-2---solution-algorithm)
    * [Hyperparameter Tuning](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#hyperparameter-tuning)
* [Algorithm Performance](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#algorithm-performance)
* [Dependencies](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution#dependencies) 

## The Competition

[RAFAEL](https://www.rafael.co.il/) (Advanced Defense Systems Ltd) 
is a well-known Israeli defense tech company ([wiki](https://en.wikipedia.org/wiki/Rafael_Advanced_Defense_Systems)).

RAFAEL constructed a [game](https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution/blob/master/Interceptor_V2.py) 
in which you play a defending turret which intercepts enemy rockets fired at two cities.  

The [RAFAEL Machine Learning Challenge 2019](http://portal.rafael.co.il/MLchallenge2019/Documents/index.html) 
competition was to write an ML-based software that plays the game independently, and gets the highest score.

## The Solution

**Approach**: breaking down the challenge into small tasks involving a single rocket \ single rocket-interceptor pair. 

**Main challenge**: correct interpretation of the rockets & interceptors lists (identifying appearance & disappearance) in the observation vector.

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

a decision-making algorithm, which chooses the highest-profile rocket (target) according to:
* **Interceptability**
* **Threat level** - hitting city\ground, time-steps to collision.
* **Relevancy** - whether the missile is already on its way to be intercepted (when an interceptor was already fired).
* **Opportunity level** - proximity to fire angle, ability to fire (when not in cooldown mode).

#### Hyperparameter Tuning

Denominator testing (with initial angle 6°)

<p float="left">
  <img src="https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution/blob/master/phase_2_solution_algorithm/results/denominator_testing/denominator_config.png" width="650">
</p>

Initial angle testing (with Denominator config 1)

<p float="left">
  <img src="https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution/blob/master/phase_2_solution_algorithm/results/initial_angle_testing/init_ang.png" width="550">
</p>

## Algorithm Performance

<p float="left">
  <img src="https://github.com/EliorBenYosef/rafael-ml-challenge-2019-winning-solution/blob/master/phase_2_solution_algorithm/results/result_video.gif" width="500">
</p>

## Dependencies
* Python 3.6.5
* Tensorflow 1.10.0
* Keras 2.2.2
* Numpy
* Matplotlib
