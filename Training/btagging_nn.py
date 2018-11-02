#!/usr/bin/env python
'''

    Script to train NN for jet flavour identification purpose (b-, c- and light jets): training and evaluation with Keras

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 ./btagging_nn.py

    Switch backend using: KERAS_BACKEND=tensorflow ./btagging_nn.py

'''
from __future__ import print_function
import numpy as np
import pandas as pd
import os, time, json, argparse
from datetime import date
from keras import backend
from keras.models import Sequential
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, History, ReduceLROnPlateau    
from keras.utils.visualize_util import plot
from keras.models import model_from_json

from keras.callbacks import TensorBoard
from btag_nn import save_history, transform_for_Keras
import h5py
from btagging_nn_models import get_seq_model
from btagging_nn_models import get_func_model

def _run():
    args = _get_args()
    np.random.seed(args.seed_nr)  # for reproducibility

    # Create directories
    os.system("mkdir -p KerasFiles/")
    os.system("mkdir -p KerasFiles/input/")
    os.system("mkdir -p KerasFiles/LogFiles/")
    protocol_dict = {}

    # Load data
    nb_classes = 2
    begin = time.time()
    X, Y, W_train, W_test, train, test, val, arr_baseline_tagger, arr_jet_pt, arr_jet_mass, arr_jet_eta = transform_for_Keras(nb_classes)
    end = time.time()
    loading_time = end - begin  # in seconds
    nb_features = X.shape[1]  # 100


    print (X.shape, W_train.shape, Y.shape)
  
    for arg in vars(args):
        protocol_dict.update({arg: getattr(args, arg)})

    # model selection (or reloading)
    if args.reload_nn[0]=='':
        if bool(args.func) == True: model, subdir_name = get_func_model(args.model, args.number_layers, nb_features, args.activation_function, args.l1, args.l2, args.activity_l1, args.activity_l2, args.init_distr, nb_classes, args.number_maxout, args.batch_normalization)
        else: model, subdir_name = get_seq_model(args.model, args.number_layers, nb_features, args.activation_function, args.l1, args.l2, args.activity_l1, args.activity_l2, args.init_distr, nb_classes, args.number_maxout, args.batch_normalization)
        subdir_name +="_"+args.optimizer+"_clipn"+str(int(args.clipnorm*100))+"_"+args.objective

        part_from_inFile = "mytest"

        os.system("mkdir -p KerasFiles/%s/" % (subdir_name))
        os.system("mkdir -p KerasFiles/%s/Keras_output/" % (subdir_name))
        os.system("mkdir -p KerasFiles/%s/Keras_callback_ModelCheckpoint/" % (subdir_name))
    else:
        # reload nn configuration from previous training
        part_from_inFile = "RE__"+args.reload_nn[0].split(".json")[0].split("__")[2]+"__"+args.reload_nn[0].split(".json")[0].split("__")[3]
        subdir_name = args.reload_nn[0].split(".json")[0].split('/')[1]
        model = model_from_json(open(args.reload_nn[0]).read())
        model.load_weights(args.reload_nn[1])

    learning_rate=args.learning_rate
    for learning_rate in [0.01]:
            print("learning_rate!", learning_rate)

    #        learning_rate=learning_rate*3
            out_str = subdir_name+part_from_inFile+"_"+backend._BACKEND+"__lr"+str(int(learning_rate*1000))+"_trainBS"+str(args.batch_size)+"_nE"+str(args.nb_epoch)+"_s"+str(args.seed_nr)+"_"+str(date.today());
            if args.LRS:
                out_str.replace("__lr"+str(int(learning_rate*1000)),"__LRS");

            if args.validation_split > 0.:
                out_str +="_valsplit"+str(int(args.validation_split*100))+"_"
            else:
                out_str+="_val"+str(int(args.validation_set*100))

            if args.clipnorm!=0.:
                if args.optimizer=="adam":
                    model_optimizer=Adam(lr=learning_rate, clipnorm=args.clipnorm)
                elif args.optimizer=="adamax":
                    model_optimizer=Adamax(lr=learning_rate, clipnorm=args.clipnorm)
                model.compile(loss=args.objective, optimizer=model_optimizer, metrics=["accuracy"])
            else:
                model.compile(loss=args.objective, optimizer=args.optimizer, metrics=["accuracy"])

            # Callback: early stopping if loss does not decrease anymore after a certain number of epochs
            early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience, mode='auto')


            # Callback: will save weights (= model checkpoint) if the validation loss reaches a new minium at the latest epoch
            if bool(args.func) == True: model_check_point = ModelCheckpoint("KerasFiles/"+subdir_name+"/Keras_callback_ModelCheckpoint/Func_weights_"+out_str+"__{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5", monitor="val_loss", verbose=1, save_best_only=True, mode="auto")
            else: model_check_point = ModelCheckpoint("KerasFiles/"+subdir_name+"/Keras_callback_ModelCheckpoint/weights_"+out_str+"__{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5", monitor="val_loss", verbose=1, save_best_only=True, mode="auto")
            
            learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, verbose = 1, factor=0.5, min_lr = 0.000001)
            print( learning_rate_reduction)

            training_time = time.time()
            callbacks = [model_check_point]
            #callbacks.append(learning_rate_reduction)
            if args.patience>=0:
                callbacks.append(early_stopping)
               
            if backend._BACKEND=="tensorflow":
                os.system("mkdir -p ./KerasFiles/TensorFlow_logs/")
                tensorboard = TensorBoard(log_dir='KerasFiles/TensorFlow_logs/'+out_str, histogram_freq=1)

                callbacks.append(tensorboard)


            if args.validation_split==0.:
                history = model.fit(X[train], Y[train],
                                    batch_size=args.batch_size, nb_epoch=args.nb_epoch,
                                    callbacks=callbacks,
                                    #show_accuracy=True,#inesochoa
                                    verbose=1,
                                    validation_data=(X[val], Y[val], W_train[val]),
                                    sample_weight=W_train[train]
                                   ) # shuffle=True (default)

            else:
                print( "test")
                history = model.fit(X[train], Y[train],
                                    batch_size=args.batch_size, nb_epoch=args.nb_epoch,
                                    callbacks=callbacks,
                                    #show_accuracy=True,#inesochoa
                                    verbose=1,
                                    validation_split=args.validation_split,
                                    sample_weight=W_train[train],
                                    shuffle=True# (default) 
                                   ) # shuffle=True (default)
            # store history:
            history_filename = "KerasFiles/"+subdir_name+"/Keras_output/hist__"+out_str+".h5"
            save_history(history, history_filename)
            training_time=time.time()-training_time

            evaluation_time = time.time()
            score = model.evaluate(X[test], Y[test], sample_weight=W_train[test], verbose=1)
                                   #show_accuracy=True,  sample_weight=sample_weights_testing, verbose=1)#inesochoa
            evaluation_time=time.time()-evaluation_time
            protocol_dict.update({"Classification score": score[0],"Classification accuracy": score[1]})

            prediction_time = time.time()
            predictions = model.predict(X[test], batch_size=args.batch_size, verbose=1) # returns predictions as numpy array
            prediction_time=time.time()-prediction_time

            timing_dict = {
                "Loading time (b-tagging data)": loading_time,
                "Training time": training_time,
                "Evaluation time": evaluation_time,
                "Prediction time": prediction_time
            }
            store_str = "KerasFiles/"+subdir_name+"/Keras_output/Keras_output__"+out_str+".h5"
            # save NN configuration architecture:
            json_string = model.to_json()
            open("KerasFiles/"+subdir_name+"/Keras_output/flavtag_model_architecture__"+out_str+".json", "w").write(json_string)
            # save NN configuration weights:
            model.save_weights("KerasFiles/"+subdir_name+"/Keras_output/flavtag_model_weights__"+out_str+".h5", overwrite=True)
            if not bool(args.func): classes = model.predict_classes(X[test], batch_size=args.batch_size)#, sample_weight=sample_weights[test]ing)
            plot(model, to_file="KerasFiles/"+subdir_name+"/Keras_output/plot__"+out_str+"model.eps")
            loss_list = history.history['loss']                                                                                                                                                                        

            protocol_dict.update(timing_dict)

            with open("KerasFiles/LogFiles/Log__"+out_str+".json", "w") as protocol_file:
                json.dump(protocol_dict, protocol_file, indent=2, sort_keys=True)

            h5f = h5py.File(store_str, 'w')
            h5f.create_dataset('Y_pred', data=predictions)
            h5f.create_dataset('Y', data=Y[test])
            if not bool(args.func):
                h5f.create_dataset('class', data=classes)
                pass
            h5f.create_dataset('X', data=X[test])
            h5f.create_dataset('W_train', data=W_train[test])
            h5f.create_dataset('W_test',  data=W_test [test])
            h5f.create_dataset('jet_pt',   data=arr_jet_pt[test])
            h5f.create_dataset('jet_mass', data=arr_jet_mass[test])
            h5f.create_dataset('jet_eta',  data=arr_jet_eta[test])
            h5f.create_dataset('baseline_tagger',data=arr_baseline_tagger[test])
            h5f.close()
            json_string = model.to_json()
            with open("KerasFiles/LogFiles/"+out_str+"model.json", "w") as json_file:
                json_file.write(json_string)
                pass

            print("Outputs:\n  --> saved history as:", history_filename, "\n  --> saved architecture as: KerasFiles/"+subdir_name+"/Keras_output/flavtag_model_architecture__"+out_str+".json\n  --> saved NN weights as: KerasFiles/"+subdir_name+"/Keras_output/flavtag_model_weights__"+out_str+".h5\n  --> saved predictions as ", store_str)


def _get_args():
    help_input_file = "Input file determining the pT and eta ranges as well as the c-fraction in the BG sample (default: %(default)s)."
    help_reload_nn = "Reload previously trained model, provide architecture (1st argument; JSON) and weights (2nd argument; HDF5) (default: %(default)s)."
    help_batch_size = "Batch size: Set the number of jets to look before updating the weights of the NN (default: %(default)s)."
    help_nb_epoch = "Set the number of epochs to train over (default: %(default)s)."
    help_init_distr = "Weight initialization distribution determines how the initial weights are set (default: %(default)s)."
    help_seed_nr = "Seed initialization (default: %(default)s)."
    help_validation_set = "Part of the training set to be used as validation set (default: %(default)s)."
    help_model = "Architecture of the NN. Due to the current code structure only a few are available for parameter search (default: %(default)s)."
    help_activation_function = "activation function (default: %(default)s)."
    help_learning_rate = "Learning rate used by the optimizer (default: %(default)s)."
    help_optimizer = "optimizer for training (default: %(default)s)."
    help_objective = "objective (or loss) function for training (default: %(default)s)."
    help_l1 = "L1 weight regularization penalty (default: %(default)s)."
    help_l2 = "L2 weight regularization penalty (default: %(default)s)."
    help_activity_l1 = "L1 activity regularization (default: %(default)s)."
    help_activity_l2 = "L2 activity regularization (default: %(default)s)."

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("-in", "--input_file", type=str,
                        default="PreparedSample__V47full_Akt4EMTo_bcujets_pTmax300GeV_TrainFrac85__b_reweighting.h5",
                        help=help_input_file)
    parser.add_argument("-r", "--reload_nn",
                        type=str, nargs='+', default=["",""],
                        help=help_reload_nn)

    parser.add_argument("-m", "--model",
                        type=str, default="Dense",
                        choices=["Dense", "Maxout_Dense"],
                        help=help_model)

    parser.add_argument("-obj", "--objective",
                        type=str, default="binary_crossentropy",
                        choices=["categorical_crossentropy", "mse","binary_crossentropy"],
                        help=help_objective)
    parser.add_argument("-o", "--optimizer",
                        type=str, default="adam",
                        choices=["adam", "adamax"],
                        help=help_optimizer)

    parser.add_argument("-lr", "--learning_rate",
                        type=float, default=0.001,
                        help=help_learning_rate)

    parser.add_argument("-l1", "--l1",
                        type=float, default=0.,
                        help=help_l1)
    parser.add_argument("-l2", "--l2",
                        type=float, default=0.,
                        help=help_l2)
    parser.add_argument("-al1", "--activity_l1",
                        type=float, default=0.,
                        help=help_activity_l1)
    parser.add_argument("-al2", "--activity_l2",
                        type=float, default=0.,
                        help=help_activity_l2)

    parser.add_argument("-af", "--activation_function",
                        type=str, default="relu",
                        choices=["relu", "tanh", "ELU", "PReLU", "SReLU"],
                        help=help_activation_function)
    parser.add_argument("-bs", "--batch_size",
                        type=int, default=80,
                        help=help_batch_size)
    parser.add_argument("-nl", "--number_layers",
                        type=int, default=5,
                        choices=[5,4,3],
                        help="number of hidden layers (default: %(default)s).")
    parser.add_argument("-nm", "--number_maxout",
                        type=int, default=0,
                        help="number of Maxout layers running in parallel")
    parser.add_argument("-p", "--patience",
                        type=int, default=20,
                        help="number of epochs witout improvement of loss before stopping")
    parser.add_argument("-ne", "--nb_epoch",
                    type=int, default=200,
                        help=help_nb_epoch)
    parser.add_argument("-id", "--init_distr",
                        type=str, default="glorot_uniform",
                        help=help_init_distr)
    parser.add_argument("-sn", "--seed_nr",
                        type=int, default=12264,
                        help=help_seed_nr)
    parser.add_argument("-vs", "--validation_set",
                        type=float, default=.1,
                        help=help_validation_set)
    parser.add_argument("-cn", "--clipnorm",
                        type=float, default=0.,
                        help="clipnorm for gradient clipping (default: %(default)s).")
    parser.add_argument("-vsp", "--validation_split",
                        type=float, default=0,
                        help="validation split using part of the training set (default: %(default)s)")
    parser.add_argument("-bn", "--batch_normalization", action="store_true",
                       help="normalize between layers for each batch")
    parser.add_argument("-lrs", "--LRS", action="store_true",
                       help="use Learning Rate Scheduler")
    parser.add_argument("-f", "--func",default=0, type=int,
                       help="use functional API instead of sequential.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    _run()
