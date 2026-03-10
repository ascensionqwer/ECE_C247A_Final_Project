---
title: FAQ
description: "All answers are endorsed by the instructor"
---

Clarification Comment: Dear class,

We would like to clarify the evaluation metric for the project. When comparing different architectures, the metric used for reporting performance should be the test CER.

The validation CER should only be used for model selection (e.g., choosing checkpoints or tuning hyperparameters). Final comparisons between models should be based on the test CER as specified in the project description.

Please let us know if you have any questions.

Best,
Yu-Wei & Kaifeng

---

Q: Do we need to explore all the suggested directions of the project to get full scores or just a few is fine?

A1: I believe you should try to explore some of the suggested directions and explain in the report your group's thought process
A2: just a few is fine

---

Q: In general, is our custom project expected to produce good results? will our grade depend on how well the model performs?

A: you have to give them a baseline to evaluate your project against, so if you have a particularly difficult project, you should explain that and have a lower baseline performance.

I would think that that's part of the grade; however, there seems to be more of an emphasis on trying different architectures and having justifications for your approaches. You can check the rubric for the final project (provided in the spec), I assume it will be similar.

---

Q: Hi,

For the ablation studies (e.g., number of channels, amount of training data, sampling rate), I'm wondering whether the trends observed under one architecture will generalize to others, for example, whether the relationship between channel count and CER holds similarly for a Transformer vs. an LSTM.

Is it acceptable to run these ablations on a single architecture (e.g., the baseline TDS or our best-performing model) and acknowledge this as a limitation in the discussion? Or are we expected to verify that the trends are consistent across multiple architectures?

Thank you.

A: I think that it’s generally acceptable to run ablations on a single baseline architecture, as long as you note in the discussion that the trends may not generalize across model families, so as not to overgeneralize. I think explaining your work and thought process. clearly is probably most important

But it also doesn’t hurt to test trends on other architectures you're considering. In HW5, I found it helpful to take a tiny subset of the data that trains very fast and sweep hyperparameters there, and the configs that looked good early tended to hold up when trained fully. Using that strategy, you can quickly sanity-check a second architecture and share some preliminary results too to be more comprehensive

---

Q: Hi,

I had a couple of clarifications regarding the final project expectations beyond what is specified in the grading rubric.

- Could you provide some guidance or examples of what a full-score (or high-scoring) project typically looks like in terms of scope, complexity, and depth?

- For performance, since no specific benchmarks are given, how should we gauge whether our results are “good enough”? Is it sufficient to show clear improvements over baselines and proper tuning, or is there an expected standard we should aim for?

- Are there any common pitfalls or reasons why projects tend to lose points in categories like creativity, insight, or performance that we should be mindful of?

A1: Yes, show your baselines and compare and contrast the performance of your various models.

A2: Our team is taking the approach of trying to make sure we demonstrate understanding of the content in class, so really exploring all the tools and techniques we learned in lecture. For us, besides the guidelines in the project document, we’re also looking to do some brief literature searches to see how models in the literature perform (ex. their performance stats) to motivate what accuracy or statistics we want to aim for in our model as well!

A3: For 3, I think in the project guidline explicitly mentioned that "you must experiment with at least 1 recurrent architecture (e.g,. RNN, LSTM, GRU). Projects that evaluate more architectures, such as RNN+CNN hybrids, transformers, or other architectures, will receive more creativity points in the rubric below." So I guess it would be helpful to try additional architectures and provide more results and analysis to earn more creativity points.

---

Q: I was trying to run training and testing for the final project in colab_setup.ipynb. I was able to run the training but had the following error while testing:

mismatched input '=' expecting <EOF> See https://hydra.cc/docs/1.2/advanced/override_grammar/basic for details Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

Anyone had same error or know how to resolve it?

A1: I think the problem is, the checkpoint path contains '=' (e.g. epoch=116-step=14040.ckpt), which Hydra misparses as a key-value separator.

I fixed this with wrapping the path in single-quotes inside double-quotes, like

"checkpoint='/your_path../checkpoints/epoch=116-step=14040.ckpt'"

A2: You may try adding a backslash(\) as an escape character before every equals sign in your checkpoint path.

---

Q: I am having some problems with setting up the conda environment as specified in the github. For some reason, when I run the create command, it takes an obscenely long time and never seems to finish. Can anyone advise?

A1: I had a similar issue, I deleted my environment and cleared my local cache which worked for me

---

Q: I tried using environment.yml and conda to create an environment but because it was taking too long, I just gave up and used pip to install requirements.txt and created a virtual env using 

`python -m venv .venv`
Is that okay?

A: Yes, you can use any python package managers of your choice.

---

Q: I got the error mentioning that No such file or directory', 
File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper File "h5py/h5f.pyx", line 102, in h5py.h5f.open

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/content/drive/MyDrive/final_project/emg2qwerty-main/data/2021-06-03-1622765527-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

Assuming this shouldn't happen?

A: Did you have all 18 hdf5 files under the data directory?

---

Q: I'm wondering what validation cer everyone is getting when running the baseline model? I get a very different validation cer at epoch 40 when running the baseline model without any changes in hyperparameters compare to the value provided in pdf

A: I stopped training at epoch 30 instead of 40 because the validation performance had already stabilized by that point.

At epoch 30, I obtained:

Validation CER ≈ 26

Test CER ≈ 26

From the training curves, the validation CER steadily decreased up to around epoch 30 and then began to plateau. The validation loss also flattened, indicating diminishing returns from continuing training. Extending training to epoch 40 did not produce meaningful improvements and introduced slightly more variance in the validation metrics, which could suggest mild overfitting or noise.

---

Q: #5 says the following: How fast does sEMG data need to be sampled for good performance? Investigate the relationship between sampling rate and CER.

When testing different sampling rates we are essentially working with less information per sample than baseline at 2000 Hz. So does this means our "in features" also need to scale down accordingly as we have less ? or am i misunderstanding what this part is asking for ? 

A1: I don't think you need to scale down the in features? I think the idea is that when you test different sampling rates, you just sample the sEMG less and keep the same time window. Essentially, you will just have fewer time samples at lower sampling rates. Empirically, you're just testing if sometimes less information could be better for performance.

think of it as you’re dropping some of the samples, like every few.

Followup: In that case what is the difference between #4 and #5 if we are just testing wether less data is better ?

A2: I think the difference is how many examples you trained on (Direction 4) versus how detailed the examples are (Direction 5). In practice, when explore Direction 4, subsamples the training set (e.g.35/50/100% of time windows) while keeping 2kHz fixed. When explore Direction 5 keep same time window with different sampling rates, then compare performance.

---

Q: Hi, for our training, should we keep the same number of training epochs, or should we keep the same early stopping rules?

A: I think keeping the same early stopping rule is generally better, since it allows us to fairly compare the best val_acc achieved by different architectures.


yes it should be fine.

---

Q: hi, when I try to run the baseline on colab, it seems like I don't get any useful information, is that correct? (They also attached an image of outputs that aren't correctly parsed due to their terminal not supporting unicode properly, but it looks like it's running fine)

A: As long as the models are saved in the logs folder and you are able to use the best one in the testing then I would imagine that an unreadable output wouldn't matter too much

thats strange, are you able to access the logs?

---

Q: do we have to finish all the trainning epoch in order to see the checkpoint? I accidentally stop trainning with several epoch, but I don't see the checkpoint file, is that normal?

A: There should be checkpoints that are saved throughout training.

---

Q: Hi, so for the already given code, I am aware that data augs and lstm encoders are given, but do we still have to code the architecture like an rnn? Also what libraries are we allowed to use?

A: You are allowed to use any libraries that you want (i.e. PyTorch, TensorFlow); of course you'll need to add the implemented architecture but you don't need to code an RNN from the ground up for example.

As said in class, you don't need to do things completely manually like in the homeworks. Since we have done it already you're free to use various libraries that already have the main architecture for things like RNNs done, you just need to add that into the emg2qwerty files.

---

Q: Hi, I noticed that the final project instructions mentioned the training and evaluation of a baseline model, which we can find in the notebook Colab setup.ipynb. I was wondering whether we would have to retrain and evaluate this model from scratch for comparison purposes or whether we can just use the reported result of validation CER = 30 in our paper.

A: We have to use our own, the professor mentioned this in class.

I suggest you try retraining the baseline model, since the result seem a bit different from that stated in the project outline. I actually obtained a validation CER of 19 after 100 epochs.

---

Q: I tried running baseline training but the validation seems to be 0%? Anyone else had issues with the data loader?

A: The actual validation CER is displayed (the part that mentions 'val/CER').

--- 

Comment: If you run training and you get an error that it cannot access the data files, it might be because you have the folder containing all the files nested within the data folder. You need to get rid of that intermediate folder.

---

Q: For people who have setup baseline with Colab, has anyone/TAs successfully ran the command:

pip install -r requirements.txt


It would be helpful to know if TAs or others were able to successfully setup their environment. Otherwise, the Colab route is not a viable option for setting up baseline.

It doesn't install all the way through.

A: start by clicking on the "Runtime version" dropdown in your settings window, which is currently set to "Latest (recommended)."

From that menu, look for and select the "Fallback runtime version" to switch your notebook to an older Python environment that is compatible with your requirements.

Once you have selected the fallback option, click "Save" to apply the changes. Finally, wait a brief moment for your Colab environment to fully restart and reconnect, and then rerun your !pip install -r requirements.txt command, which should now bypass the build error and successfully install your packages.

---

Q: Hi my colab accidentally stopped at about just over a hundred epochs and so I loaded in a checkpoint but it doesn’t actually stop at 150, it goes a bit above cuz the new one starts at 0 epochs but takes into account the checkpoint details, so I stopped it when it went over 150 epochs instead of exactly 150. Is this ok?

A: Yes, this is completely okay, because when you loaded your Checkpoint (e.g., checkpoint.pth), the model successfully resumed learning with all its preserved weights intact.

The reset epoch counter is just a logging detail, so manually stopping it slightly over your target of 150 total passes is entirely harmless. A few extra cycles over the data won't ruin your progress unless the model was already Overfitting (e.g., Train Loss: 0.05 | Val Loss: 1.85).

---

Q: I tried a CNN + Transformer model and I found that there is a very huge gap between the validation cer and test cer. I thought it was overfitting but the validation cer is not very high, so I'm curious what problem might I encounter. I have a 3 layer 1D CNN and 2 transformer layers with 6 heads each

A1: Is something going wrong with the way you initialized it/structured your files? This doesn’t seem like a model problem to me
A2: I used a cnn + gru and there is no significant gap between best case val cer and test cer

---

Q: How many epochs yall typically stop at when training for the baseline model?

A1: I've been using 100 - 150 depending on when I see overfitting occur.

A2: The code professor provided force stopped at 150. Our experiment showed CER reduction until 119th epoch and stop improving.

---

Q: I have been experimenting with several data augmentation techniques using the baseline model, but so far none of them have improved the testing CER.

For milestone #2, would it still be appropriate to report these experiments even if they do not outperform the original baseline results?

Also, would it be better to evaluate the augmentation methods using our best-performing architecture instead of the baseline model?

Our group only has two members and limited computational resources, so we are trying to decide how to prioritize our experiments.

A1: While you want to focus your experiments in directions that improve test CER, it is perfectly okay to report on these other experiments that don't outperform the baseline. The project requirements even ask that you explain whether some changes didn't improve CER (or even made it worse), and why you think that is.

A2: I think it’s still appropriate to report those experiments. Negative results are still valuable.

Our group only test augmentation on the best-performing architecture due to the concern of time and limited computational budget.

---

Q: Our group implemented a Transformer-based model as an enhanced architecture using the provided baseline framework. During training and validation, the model converges normally and achieves reasonable validation performance. For example, our final validation result is approximately:

val CER ≈ 14.7

However, when running the default test pipeline, we observed an abnormal result:

test CER = 100

test IER = 100

test DER = 0

test SER = 0

This pattern suggests that the model produces extremely long predictions with excessive insertions, rather than failing completely.

After debugging, we realized the key difference between training/validation and testing:

Train / Val: use windowed inputs (window_length = 8000, padding applied)

Test: feeds the entire EMG session at once (window_length=None)

This means the Transformer is trained on relatively short windows but evaluated on very long sequences (~100k+ timesteps), which may lead to unstable decoding with greedy CTC.

Modification we tested
To diagnose this issue, we temporarily modified the test dataset construction so that test uses the same windowing scheme as train/val.

Original test setup

```
WindowedEMGDataset(
    hdf5_path,
    transform=self.test_transform,
    window_length=None,
    padding=(0, 0),
    jitter=False,
)
```

Modified test setup

```
WindowedEMGDataset(
    hdf5_path,
    transform=self.test_transform,
    window_length=self.window_length,
    padding=self.padding,
    jitter=False,
)
```

After this change, the results became consistent with validation:

val CER ≈ 14.7

test CER ≈ 16.6

This suggests that the issue is likely related to whole-session inference for Transformer models, rather than a training or model implementation bug.

Question
Would it be acceptable to evaluate our Transformer model using windowed test inference (matching the training distribution), or should we instead keep the whole-session testing protocol and implement a chunked inference strategy?

We’d appreciate any guidance on what evaluation setup would be considered most appropriate.

Thanks!

A1: I think it is not necessary to use some models with lots of parameters like transformer to get a good result. Our group are able to get a test cer of about 16.2 using the original setup with some rnn based model

A2: Yes, you can set the window_length. However, you should make it consistent across all your experiments for fair comparisons, and explain this modification in the report.

---

Q: Hello, if we possibly change the data augmentation, model architecture and hyperparameter tuning, how many epochs are we supposed to train and compare with? It's quite costly to train the entire 150 epochs for all variants of pipeline

A: This is left up to you. The main reason for considering additional epochs is if that improves your performance, as it's just a hyperparameter. If you don't want to take that full cost, I suggest you run one iteration of your final model with the maximum number of epochs (maybe increments of 50) just to see if that helps.

i think my team is going to just do the 150 for all, but it is your choice.

---

Q: For the project, my team and I decided to split up work by having each person train and evaluate one different model. This means we'll have different data augmentation techniques and approached. When comparing model performance is it fine that we have different data preprocessing techniques, or does it all have to be the same?

A1: Your group’s innovation can occur in different parts of the pipeline. For example, if your goal is to compare different model architectures, the most controlled (and fair) approach would be to keep the rest of the pipeline fixed (for example  preprocessing and data augmentation) and evaluate which model performs best. That said, in practice different architectures may benefit from slightly different preprocessing or augmentation choices, so some variation can be reasonable.

TL;DR: Ideally keep preprocessing consistent when comparing models, but it’s acceptable to make reasonable adjustments given your compute and time constraints.

Hope this clarifies!

A2: In my opinion, I would be sharing your data augmentations with your team so you can consider using at least some of the same techniques (when applicable). I think the analysis part of the process is greatly enhanced by analyzing the same data augmentation over several architectures, both as a control for the evaluation architectures and to see when the data manipulation technique is most effective.

---

Q: Did you guys do bidirectional or vanilla RNN?

A: You can choose any I guess I’m doing lstm

it’s totally up to your team!

---

Q: The accuracy is very low for the vanilla RNN. Are u guys getting the same result?

A: The professor suggested going straight to a LSTM or GRU model during lecture due to the issue of vanishing gradients.

---

Q: For number 4 in "baseline" project suggested directions. Do we tune the number of training data on every architecture that we use? Or, we can just use one model to do the analysis?

A: My team decided on a single model architecture and conducting ablation experiments, including those data sizes. I believe the experiments will yield fairly clear results, particularly regarding data sizes.

---

Q: Do we need to experiment with implementing VGGNet/ResNet/GoogleNet for full credit on the final project or would it be sufficient to use the given CNN encoder and do a hybrid with RNN/LSTM/Transformer?

A: No, almost no one is doing VGGNet/ ResNet stuff. What you’re doing is fine.

---

Q: Has anyone successfully run training with ctc_beam, not ctc_greedy? 
I got kicked out by colab twice already and am wondering what configuration values should be used to make it work.

A: I think you can try the ctc_beam since it is given. However, based on my experience, it will go much slower than ctc_greedy.

---

Q: Our group is thinking of doing TCN for one of the architectures that we will be testing on. Will that suffice for the creativity part of the project?

A: Trying different architectures like RNN, TCN, transformers, state space models etc. or different methods like data preprocessing and data augmentation all count towards creativity/diversity, as long as you are not just trying to improve your model by just tuning hyperparameters

---

Q: What would be considered a high performance model? For context, our group's best models are hitting ~16-17 for test CER. Just trying to gauge the room for improvement. Thank you!

A1: Our best is about the same as well.

A2: We reached 15.6 Val/CER over 50 epochs

---

Q: If we tried multiple models for our project, I'm wondering do we need to make number of parameters similar to each other in order to make them comparable? For example I have about 12.8M parameters for my cnn + gru model, do I need to have similar number of parameters for my cnn + lstm?

A: You should try comparing architectures with their best performing hyperparameters.

---

Q: My best model performance has test cer of 16.02, and my second best model performance has test cer of 16.2. I used two different models with same number of epochs and same seed, and there is about 1.2M number of parameters difference between models (both have 10M+ parameters). So I'm wondering if it is worth to analyze why the best model outperforms the second best model given the gap between test cers are pretty small?

A: This is a great point! If your best model has 1.2M more parameters, yet it produces a result that's just 0.08 more than the second best, if you have time and compute, I'd try running the exact same experiments with a different random seed. This might produce different results. If not, the second model should be the better choice.

---

Q: I ran this: python -m emg2qwerty.train --multirun \
  user="single_user" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=50

Which is taken from the testing part of the colab. Is this suppose to also run the test? Another person in my group didn't have this happen, and I was wondering why did it run both the validation and the test as I got the results for both.

A: Yes, it will run the test after 50 epochs training. You'll see it at last. Run the validation because you can't use test set to measure model capacity. In 50 epochs you'll get 50 models and you'll need the best model among them to test the test set. It will select a model's checkpoint on validation set to tell you the best model's performance and why it's selected and run it on test set to tell you the real capacity of this model.

---

Q: I am just now realizing that I am missing the metadata.csv file mentioned in the ReadMe. I downloaded the patient data directly from the folder here:
https://ucla.app.box.com/s/3xc4nwpfjfpo6ydjs94t0v2kuq37d5eg

Where can I find the metadata file? I have been unable to extract it alone from the aws database so far.

A1: You don't need to have the metadata file. The READ.ME is from the Meta researchers so you don't have to follow it completely.
A2: I don't have a metadata file either.. and I'm able to run fine.

---
Q: For the default project, 5 directions are suggested. Any minimum requirement of how many directions we needed to explore?

A: See earlier answer. only a few is fine

---

Q: Would it be okay to try different model architectures/hyperparams using a lower number of epochs to reduce compute (e.g. 40 epochs), then using the model that gives the highest CER, fine-tune the hyperparams such as increase the number of epochs to get the lowest possible test CER for that architecture?

A: That's pretty much the same our team is doing, find some promising architectures or preprocessing, optimizing methods first by 40 epochs, extend to 150epochs for good signaling ones.

Our team has been using 150 for everything, and kinda regret it. It took the whole day to run like 5 models, so what you're doing is better to be honest.

---

Q: Is it possible to only evaluate 50 epochs for all directions? if not, can we run 150 epochs for the final architecture?

A: You can use 50 epochs. It is up to your group.

---

Q: I just wish to confirm that can we get full credit for creativity and diversity by just trying different models? (For example, cnn + rnn (variants), transformer, TCN, etc.)

A: Yeah I’m doing custom project and the prof told me to show the baseline tho. Make sure to state it and compare with different models

what youre doing sounds fine.

---

Q: Hello, i saw someone else post about it but I am also having issue where Val CER is relatively low after 40 epochs (around 30-40) but test CER is high 90s or 100. Wanted to see if others found out about potential causes for this. 


just for insight the baseline github repo run i got around 24 val and test CER so think that was fine, but I tried adding also a biGRU with 128 hid dem and 1 layer and testing that now. I had two layers at 256 so was thinking maybe it was just too many parameters and overfitting. Unless it is just an issue with the code and how it process the test set

A: Can you try training for more epochs? It is a little strange but I think you’ll see better results with closer to 100 epochs

This is kind of related to overfitting and the problem should be fixed if you have the same setup for testing and validating (i.e. window, padding, etc. which can be found in lightning.py). But in order to have fair comparison, you need to adapt the new testing setup to all of your approaches including the baseline

---

Q: Hi I just read in the spec that they tested for 40 epochs and I tested for 150 epochs. Do I also need to test for 40 epichs to see if val cerr is around 30? With 150 it is around 20.

Thanks.

A1: You should have the validation CER for every epoch, so if you want to check the CER at epoch 40, just look at the logs. It should be around 25.

Also, the CER value mentioned in the PDF was just a rough range to give you some insight. What you should really focus on improving is the test CER.

A2: You get to define your own baseline, if you have been using 150 then that CER can serve as your baseline. But running 150 epoch could have compute time/resources implication. Also a low baseline leaves less room for improvement.

---

Q: For these directions if we want to explore them, if our group trained multiple models to try to find the best one do we have to also test for all of these models these directions?

A: My team is just using our best model

Edit (not original submitter): I believe it will count more for creativity if you test different models in different directions and compare how well they do against one another. However, given feasibility and timing, it might just be better to use the best model and explore the other directions. Both should be okay.

---

Q: Which functions/classes do we edit for parts 3,4,5 for directions? These are relations between electrode channels, training data, sampling rate, compared to val cer, respectively.

A1: For some of the directions in the spec you have to add additional functions but others can be experimented with through the yaml config files or by adding additional/changing parameters for the train and test functions provided to us in the .ipynb

A2: I edited the base.yaml, log_spectrogram.yaml, and added some python files for my experiments.

Followup: do we have to both train and test with the changed parts?

A3: yes

---

Q: I saw in the project spec that it mentions testing different data preprocessing techniques and data augmentation techniques. Do we have to do all of these, or do we choose a subset of these to do?

Please let me know,

Thanks!

Anagh

A: You can choose. You don’t have to do data augmentation at all if you don’t want to either.

There was an earlier question about this, and you only need to do a few to get full credit for the creativity aspect 

---