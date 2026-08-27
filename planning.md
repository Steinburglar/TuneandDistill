User interface should center on specifying the “run[train, val, test, sample, distill]” line
Student can be similar to model, but by default be from scratch, and match the foundation model unless otherwise specified.
Sampler will require the most design, most open ended. How should it know which model to use in a script
Separate labelling from sampling?


Steps:
finetune/train FM
Sample frames (with or without FM)
Label frames synthetically (with FM)
distill/train student model

Problems:
How to handle restarts? Restarts are meaningful if there is just one model. Becomes unclear if there are multiple.

In order to generate frames/labels, do we have to compile the teacher? Should we even be doing compiling in the training script.

Possible solutions:
Separate into two separate configs, a teacher and student config, provide new command line script that manages things on top. “nequip-PFD”. Then need config for sampling procedure as well
All in one config, make checkpoint refer to teacher, need “student_checkpoint” for student. All train, val, test, of student are just stu_train, stu_val, stu_test. Student config is inherited from teacher config except for checkpoint path and where otherwise specified?
Distinction to draw: two slightly different things
What is specified in what config file
Which processes are called in the train() script, or run in a higher level script that runs train() or a wrapped version of train on a config, then does other things (like sampling/labeling frames) 


So, we can have 1 config with the specifications, but have a script that calls out the train, package, and compile scripts separately, and does not call one from the other.






Final Plan:

Separate Finetuning from Distillation. Finetuning is largely written already, just need to make sure the modifiers make sense.

Instead of the primary goal being a full finetune then distill script, we just want to offer a new distillation script, nequip-distill.

What you need for nequip-distill
A teacher model artifact and a compatible ase calculator; (ideally compiled/fast).
Frame(s) that you would like to either run MD from, or jiggle or something to sample new ones.
A normal config for the student model you want (same requirements as nequip train())
Plus: a new sampling config section that clarifies sampling procedure.
Path and calculator type of the teacher model


Given a compiled, packaged, or otherwise ase calculator compatible artifact, you simply run nequip-distill –cn distil.yaml, and it will do and save the sampling, then do a normal training for you.

Restart handling:

Should be handled similar to nequip-train. If you have an existing half or fully finished dataset, you can hand samples_path, and it will resume from there. 

Similarly, once you enter training, if it sees a checkpoint path in the config, it will do the normal nequip train() behavior with that sample.
