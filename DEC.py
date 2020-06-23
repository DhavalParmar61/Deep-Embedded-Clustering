import sys
import numpy as np
import keras.backend as K
import pickle
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Input
from keras.initializers import RandomNormal
from keras.engine.topology import Layer, InputSpec
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import normalize
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


class FinalClusteringLayer(Layer):
    def __init__(self, output_dim, input_dim=None, weights=None, alpha=1.0, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(FinalClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, input_dim))]
        self.W = K.variable(self.initial_weights)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        q = 1.0 / (1.0 + K.sqrt(K.sum(K.square(K.expand_dims(x, 1) - self.W), axis=2)) ** 2 / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q


class Model(object):
    def __init__(self, n_clusters, input_dim, alpha=1.0, batch_size=256, **kwargs):

        super(Model, self).__init__()

        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = 0.1
        self.iters_lr_update = 20000
        self.lr_change_rate = 0.1
        dropout_fraction = 0.2
        init_stddev = 0.01

        # Layer-wise training
        self.encoders_dims = [self.input_dim, 500, 500, 2000, 10]
        self.input_layer = Input(shape=(self.input_dim,), name='input')

        self.layer_wise_autoencoders = []
        self.encoders = []
        self.decoders = []
        for i in range(1, len(self.encoders_dims)):
            encoder_activation = 'linear' if i == (len(self.encoders_dims) - 1) else 'relu'
            encoder = Dense(self.encoders_dims[i], activation=encoder_activation,
                            input_shape=(self.encoders_dims[i - 1],),
                            kernel_initializer=RandomNormal(mean=0.0, stddev=init_stddev, seed=None),
                            bias_initializer='zeros', name='encoder_dense_%d' % i)
            self.encoders.append(encoder)

            decoder_index = len(self.encoders_dims) - i
            decoder_activation = 'linear' if i == 1 else 'relu'
            decoder = Dense(self.encoders_dims[i - 1], activation=decoder_activation,
                            kernel_initializer=RandomNormal(mean=0.0, stddev=init_stddev, seed=None),
                            bias_initializer='zeros',
                            name='decoder_dense_%d' % decoder_index)
            self.decoders.append(decoder)

            autoencoder = Sequential([
                Dropout(dropout_fraction, input_shape=(self.encoders_dims[i - 1],), name='encoder_dropout_%d' % i),
                encoder,
                Dropout(dropout_fraction, name='decoder_dropout_%d' % decoder_index),
                decoder
            ])
            autoencoder.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))
            self.layer_wise_autoencoders.append(autoencoder)

        # build the end-to-end autoencoder for finetuning
        self.encoder = Sequential(self.encoders)
        self.encoder.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))
        self.decoders.reverse()
        self.autoencoder = Sequential(self.encoders + self.decoders)
        self.autoencoder.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))

    def p_mat(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def initialize(self, X, save_autoencoder=False, layerwise_pretrain_iters=50000, finetune_iters=100000):

        iters_per_epoch = int(len(X) / self.batch_size)
        layerwise_epochs = max(int(layerwise_pretrain_iters / iters_per_epoch), 1)
        finetune_epochs = max(int(finetune_iters / iters_per_epoch), 1)

        print('layerwise pretrain')
        current_input = X
        lr_epoch_update = max(1, self.iters_lr_update / float(iters_per_epoch))

        def step_decay(epoch):
            initial_rate = self.learning_rate
            factor = int(epoch / lr_epoch_update)
            lr = initial_rate / (10 ** factor)
            return lr

        lr_schedule = LearningRateScheduler(step_decay)

        for i, autoencoder in enumerate(self.layer_wise_autoencoders):
            if i > 0:
                weights = self.encoders[i - 1].get_weights()
                dense_layer = Dense(self.encoders_dims[i], input_shape=(current_input.shape[1],), activation='relu',
                                    weights=weights)
                encoder_model = Sequential([dense_layer])
                encoder_model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate, decay=0, momentum=0.9))
                current_input = encoder_model.predict(current_input)

            autoencoder.fit(current_input, current_input, batch_size=self.batch_size, epochs=layerwise_epochs,
                            callbacks=[lr_schedule])
            self.autoencoder.layers[i].set_weights(autoencoder.layers[1].get_weights())
            self.autoencoder.layers[len(self.autoencoder.layers) - i - 1].set_weights(
                autoencoder.layers[-1].get_weights())

        print('Finetuning autoencoder')
        self.autoencoder.fit(X, X, batch_size=self.batch_size, epochs=finetune_epochs, callbacks=[lr_schedule])

        if save_autoencoder:
            self.autoencoder.save_weights('autoencoder.h5')

        # update encoder weights
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].set_weights(self.autoencoder.layers[i].get_weights())

        # initialize cluster centres using k-means
        print('Initializing cluster centres with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(X))
        self.cluster_centres = kmeans.cluster_centers_

        self.DEC = Sequential([self.encoder, FinalClusteringLayer(self.n_clusters, weights=self.cluster_centres)])
        self.DEC.compile(loss='kullback_leibler_divergence', optimizer='adadelta')
        return

    def cluster_acc(self, y_true, y_pred):
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, w

    def pred_cluster(self, X, y=None, tol=0.01, iter_max=3 * 1e4, **kwargs):

        update_interval = int(X.shape[0] / self.batch_size)
        save_interval = int(X.shape[0] / self.batch_size * 50)

        train = True
        iteration, index = 0, 0
        self.accuracy = []
        while train:
            sys.stdout.write('\r')
            if iter_max < iteration:
                print('Reached maximum iteration limit. Generating image of Feature cluster using t-SNE.')
                tsne = TSNE(n_components=2, random_state=0)
                z_pres = tsne.fit_transform(z)
                target_ids = range(10)
                plt.figure(figsize=(6, 5))
                colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'magenta', 'orange', 'purple'
                for i, c in zip(target_ids, colors):
                    plt.scatter(z_pres[y == i, 0], z_pres[y == i, 1], c=c, label=i, s=0.05)
                plt.legend()
                plt.savefig(f'Learned_feature.png')
                return self.y_pred

            # from DEC model to encoder.
            if iteration % update_interval == 0:
                self.q = self.DEC.predict(X, verbose=0)
                self.p = self.p_mat(self.q)

                y_pred = self.q.argmax(1)
                delta_label = ((y_pred == self.y_pred).sum().astype(np.float32) / y_pred.shape[0])
                if y is not None:
                    acc = self.cluster_acc(y, y_pred)[0]
                    self.accuracy.append(acc)
                    print('Iteration ' + str(iteration) + ', Accuracy ' + str(np.round(acc, 5)))
                else:
                    print(str(np.round(delta_label * 100, 5)) + '% change in label assignment')

                if delta_label < tol:
                    print('Reached tolerance threshold.')
                    train = False
                    continue
                else:
                    self.y_pred = y_pred

                for i in range(len(self.encoder.layers)):
                    self.encoder.layers[i].set_weights(self.DEC.layers[0].layers[i].get_weights())
                self.cluster_centres = self.DEC.layers[-1].get_weights()[0]

            # train on batch
            sys.stdout.write('Iteration %d, ' % iteration)
            if (index + 1) * self.batch_size > X.shape[0]:
                loss = self.DEC.train_on_batch(X[index * self.batch_size::], self.p[index * self.batch_size::])
                index = 0
                sys.stdout.write('Loss %f' % loss)
            else:
                loss = self.DEC.train_on_batch(X[index * self.batch_size:(index + 1) * self.batch_size],
                                               self.p[index * self.batch_size:(index + 1) * self.batch_size])
                sys.stdout.write('Loss %f' % loss)
                index += 1

            if iteration % save_interval == 0:
                z = self.encoder.predict(X)
                z_np = np.asarray(z)
                np.save(f'z_{iteration}', z_np)
                self.DEC.save('DEC_model_' + str(iteration) + '.h5')

            iteration += 1
            sys.stdout.flush()
        return


if __name__ == "__main__":
    np.random.seed(1234)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_all = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)
    X = x_all.reshape(-1, x_all.shape[1] * x_all.shape[2])
    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32) * 0.02
    Y = Y[p]
    np.save('Y', Y)

    cluster_model = Model(n_clusters=10, input_dim=784)
    cluster_model.initialize(X, finetune_iters=60000, layerwise_pretrain_iters=30000)
    cluster_model.pred_cluster(X, y=Y)