# Reinforcement learning a transfer or active learning policy 

[![Build Status](https://travis-ci.com/pblankley/thesis.svg?token=FWBabyaZZecFnSgMiD6n&branch=master)](https://travis-ci.com/pblankley/thesis)

### The Idea 

The main idea is that we have two datasets where one is a "source" and one is a "target."  The source has plenty of labeled points, while the target has very few labeled points.  The problem is we want a classifier that will perform well on the target, which is similar to the source but we do not have enough training points in the target for a easy transfer learning application.  We want to see if we can use reinforcement learning to learn the best time to get more points in the target (i.e. have experts label the points), and find the time that we have enough points in the target to use transfer learning effectively on the data.  The goal is to minimize the number of points we gather from the target, because this is expensive, while producing the best possible accuracy for a given cost of obtaining points in the target. 

### Step One 

The first thing I tried was simply implementing a several common policies and finding the best one for my test data (MNIST).  I implemented a boundary, maximum entropy, least confidence, uniform and random policy. Then I ran each of these policies several times over the dataset to have a distribution of results I could compare and see which policies perform the best generally. 

The results of these experiemnts can be found in the folder reports/figures/ and they show the boundary policy as being consistently the best policy, with the other active policies (max entropy, uniform,  and least confidence) approaching boundary as the number of epochs increases.  The random policy, as expected, is significantly worse performing than the active policies as the number of epochs increases.

### Step Two

The next step I took was to reinforcement learn the best active learning policy for the unlabeled dataset.  This is a ill-posed problem, however, because in step one I saw that the boundary policy was pretty much exculsively the best performing policy.  So, I would expect the reinforcement learner, if it is working correctly, to find a meta policy where it just picks the boundary policy.  This is the expected result, and after enough training the reinforcement learner (not surprisingly) finds this result.  It is pretty consistent with finding this result. 

### Step Three

The next step in the process is to reinforcement learn the best policy between a transfer learning policy and an active learning policy (which I chose deterministically as the boundary policy for simplicity). The reinforcement learner has two choices for its policy then.  It  can either choose the active learning policy and train for k epochs on the target set with the new points, or it can train for e epochs on the full source dataset and then train for k epochs on the target dataset.  Now, this is fundamentally not a good setup because it will almost ALWAYS be better to pick new points in the target when you are measuring success on accuracy in the target.  However, we are trying to find the point where its a smarter idea to train on the transfer learner for a given cost of obtaining new points in the target.  In my current implementation I just have a scalar penalty to the reward for getting points in the target.  

The other big problem with this is consistently training on the source and then going back to the target, especially if e > k, will produce a model that generally performs better on the source and doesn't really help performance on the target.  This is one of the big issues with how I am currently doing things.

### Future Work

I will no longer be able to give any time to this project, so it is going to have to sit in a state of partial completion. The next step in my mind is to restructure the problem itself.  I think the above problem is kind of ill-posed, especially for a reinforcement learner that is incentivised to learn just one policy most of the time. The better way to formulate it (I think) is to have a "budget" you are willing to spend and you record cost for getting new points on the target, and a very minute cost for training on the source.  You then can try to maximize accuracy for a given "budget."  This one runs into a little of the same problem above though, but a little less so because it constrains the total cost.  Another way to restructure it is to have a goal accuracy and try to get to that accuracy with minimal cost.  That is the coolest reformulation, in my mind.  It is cool because it might actually show you that once you get n points in the target you can just barely get to the accuracy you want by training the transfer learner more at minimal cost.  This is the sort of interesting result I would like to see. 


