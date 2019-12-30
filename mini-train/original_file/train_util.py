import uuid
import sys
sys.path.append('..')
import numpy as np
from time import sleep
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from deepmoji.global_variables import (
    FINETUNING_METHODS,
    WEIGHTS_DIR)
from deepmoji.finetuning import (
    freeze_layers,
    sampling_generator,
    train_by_chain_thaw,
    find_f1_threshold)

def relabel(y, current_label_nr, nb_classes):
    if nb_classes == 2 and len(y.shape) == 1:
        return y

    y_new = np.zeros(len(y))
    y_cut = y[:, current_label_nr]
    label_pos = np.where(y_cut == 1)[0]
    y_new[label_pos] = 1
    return y_new

def prepare_labels(y_train, y_val, y_test, iter_i, nb_classes):
    # Relabel into binary classification
    y_train_new = relabel(y_train, iter_i, nb_classes)
    y_val_new = relabel(y_val, iter_i, nb_classes)
    y_test_new = relabel(y_test, iter_i, nb_classes)
    return y_train_new, y_val_new, y_test_new


def prepare_generators(X_train, y_train_new, X_val, y_val_new, batch_size, epoch_size):
    # Create sample generators
    # Make a fixed validation set to avoid fluctuations in validation
    train_gen = sampling_generator(X_train, y_train_new, batch_size,
                                   upsample=True)
    val_gen = sampling_generator(X_val, y_val_new,
                                 epoch_size, upsample=True)
    X_val_resamp, y_val_resamp = next(val_gen)
    return train_gen, X_val_resamp, y_val_resamp


def finetuning_callback(checkpoint_path, patience, verbose,savepath):
    cb_verbose = (verbose >= 2)
    checkpointer = ModelCheckpoint(monitor='val_loss', filepath=checkpoint_path,
                                   save_best_only=True, verbose=cb_verbose)
    checkpointer2=ModelCheckpoint(monitor='val_loss', filepath='{}/model.hdf5'.format(savepath),
                                   save_best_only=True, verbose=cb_verbose)
    # earlystop = EarlyStopping(monitor='val_loss', patience=patience,
    #                           verbose=cb_verbose)
    return [checkpointer,checkpointer2,TensorBoard(log_dir='{}/mytensorboard'.format(savepath))]

def class_trainable(model, nb_classes, train, val, test, epoch_size,
                             nb_epochs, batch_size, init_weight_path,
                             checkpoint_weight_path,savepath, patience=5,
                             verbose=True):
    total_f1 = 0
    nb_iter = nb_classes if nb_classes > 2 else 1

    # Unpack args
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    # Save and reload initial weights after running for
    # each class to avoid learning across classes
    model.save_weights(init_weight_path)
    for i in range(nb_iter):
        if verbose:
            print('Iteration number {}/{}'.format(i + 1, nb_iter))

        model.load_weights(init_weight_path, by_name=False)
        y_train_new, y_val_new, y_test_new = prepare_labels(y_train, y_val,
                                                            y_test, i, nb_classes)
        train_gen, X_val_resamp, y_val_resamp = \
            prepare_generators(X_train, y_train_new, X_val, y_val_new,
                               batch_size, epoch_size)

        if verbose:
            print("Training..")
        callbacks = finetuning_callback(checkpoint_weight_path, patience, verbose=2,savepath=savepath)
        steps = int(epoch_size / batch_size)
        model.fit_generator(train_gen, steps_per_epoch=steps,
                            max_q_size=2, epochs=nb_epochs,
                            validation_data=(X_val_resamp, y_val_resamp),
                            callbacks=callbacks, verbose=0)

        # Reload the best weights found to avoid overfitting
        # Wait a bit to allow proper closing of weights file
        sleep(1)
        model.load_weights(checkpoint_weight_path, by_name=False)

        # Evaluate
        y_pred_val = np.array(model.predict(X_val, batch_size=batch_size))
        y_pred_test = np.array(model.predict(X_test, batch_size=batch_size))

        f1_test, best_t = find_f1_threshold(y_val_new, y_pred_val,
                                            y_test_new, y_pred_test)
        if verbose:
            print('f1_test: {}'.format(f1_test))
            print('best_t:  {}'.format(best_t))
        total_f1 += f1_test

    return total_f1 / nb_iter


def class_train(model, texts, labels, nb_classes, batch_size,
                       method, savepath,epoch_size=64,
                       nb_epochs=1000, error_checking=True,
                       verbose=True):

    (X_train, y_train) = (texts[0], labels[0])
    (X_val, y_val) = (texts[1], labels[1])
    (X_test, y_test) = (texts[2], labels[2])

    checkpoint_path = '{}/deepmoji-checkpoint-{}.hdf5' \
                      .format(WEIGHTS_DIR, str(uuid.uuid4()))

    f1_init_path = '{}/deepmoji-f1-init-{}.hdf5' \
                   .format(WEIGHTS_DIR, str(uuid.uuid4()))

    # Check dimension of labels
    if error_checking:
        # Binary classification has two classes but one value
        expected_shape = 1 if nb_classes == 2 else nb_classes

        for ls in [y_train, y_val, y_test]:
            if len(ls.shape) <= 1 or not ls.shape[1] == expected_shape:
                print('WARNING (class_avg_tune_trainable): '
                      'The dimension of the provided '
                      'labels do not match the expected value. '
                      'Expected: {}, actual: {}'
                      .format(expected_shape, ls.shape[1]))
                break

    lr = 0.001

    loss = 'binary_crossentropy'


    # Compile model
    adam = Adam(clipnorm=1, lr=lr)
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])

    # Training
    if verbose:
        print('Method:  {}'.format(method))
        print('Classes: {}'.format(nb_classes))


    result = class_trainable(model, nb_classes=nb_classes,
                                          train=(X_train, y_train),
                                          val=(X_val, y_val),
                                          test=(X_test, y_test),
                                          epoch_size=epoch_size,
                                          nb_epochs=nb_epochs,
                                          batch_size=batch_size,
                                          init_weight_path=f1_init_path,
                                          checkpoint_weight_path=checkpoint_path,
                                          savepath=savepath,
                                          verbose=verbose)
    return model, result


