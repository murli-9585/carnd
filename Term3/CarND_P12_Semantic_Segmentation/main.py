import os.path
import helper
import tensorflow as tf
import time
import warnings
from distutils.version import LooseVersion

import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), ("Please use "
        "TensorFlow version 1.0 or newer.  "
        "You are using {}'.format(tf.__version__)")
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn("No GPU found. Please use a GPU to train "
            "your neural network.")
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/"
        and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob,
        layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.
        Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Ref walkthrough.
    # Add a 1x1 conv layer to layer7 of Vgg with kernel size
    # FCN layer 7.
    fcn_l7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # De_Convulte layer 7.
    # Add deconvolution layer & set kernel size of 4x4 & stride of 2x2 to upsample
    fcn_dconv_l7 = tf.layers.conv2d_transpose(fcn_l7, num_classes, 4, 2,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Add a 1x1 conv layer to layer4 of Vgg with kernel size of 1 & stride of 1
    # FCN layer 4.
    fcn_l4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Skip layer 4.
    skip_l4 = tf.add(fcn_dconv_l7, fcn_l4)

    # Add deconvolution layer & set kernel size of 4x4 & stride of 2x2.
    # De_convolute layer 4.
    fcn_dconv_l4 = tf.layers.conv2d_transpose(skip_l4, num_classes, 4, 2,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Add a 1x1 conv layer to layer3 of Vgg with kernel size of 1 & stride of 1
    # FCN layer 3.
    fcn_l3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Skip connection; layer 3.
    skip_l3 = tf.add(fcn_dconv_l4, fcn_l3)

    # Add deconvolution layer & set kernel size of 16x16 & stride of 8x8.
    # De_convolute layer 3.
    fcn_dconv_l3 = tf.layers.conv2d_transpose(skip_l3, num_classes, 16, 8,
            padding='same',
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # tf.Print(output, [tf.shape(output)[1:3]])
    return fcn_dconv_l3

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # make logits a 2D tensor where each row represents a pixel and
    # each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # Calculate cross entropy loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
            (logits = logits, labels = correct_label))

    # Use adam optimizer
    #https://stats.stackexchange.com/questions/184448/difference-between-gradientdescentoptimizer-and-adamoptimizer-tensorflow
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_operation = optimizer.minimize(cross_entropy_loss)

    return logits, train_operation, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
        cross_entropy_loss, input_image, correct_label, keep_prob,
        learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.
        Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    keep_prob_val = 0.75
    learn_rate_val = 0.0001
    result_file = open("result.txt", "a")
    result_file.write("Training Results... \n")

    print("Start Training .. ")
    print()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            start_time = time.time()
            for image, label in get_batches_fn(batch_size):

                #training
                _, loss = sess.run([train_op, cross_entropy_loss],
                        feed_dict={input_image: image, correct_label: label,
                        learning_rate: learn_rate_val, keep_prob: keep_prob_val})

            end_time = time.time()
            epoch_train_time = end_time - start_time
            msg = "Epoch: {0}/{1} || Execution Time: {2} seconds || Loss: {3} \n".format(
                    epoch, epochs, epoch_train_time, loss)
            print(msg)
            result_file.write(msg)

    result_file.close()

tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of
    # the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir,
                'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # MM TODO: ATM, not tested on GPU, so time to test is low.
        epochs = 20 # 16, 20, 30
        batch_size = 16

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(
                sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        #TF placeholders
        correct_label = tf.placeholder(dtype=tf.float32,
                shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(dtype=tf.float32)

        #call optimizer gives logits
        logits, train_operation, cross_entropy_loss = optimize(layer_output,
                correct_label, learning_rate, num_classes)

        train_nn(sess, epochs, batch_size, get_batches_fn, train_operation,
                cross_entropy_loss, input_image, correct_label, keep_prob,
                learning_rate)

        # Save images.
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                logits, keep_prob, input_image)

        """
        # Process video of trained model.
        def to_video(original_image, sess=sess, image_shape=image_shape,
                logits=logits, keep_prob=keep_prob, image_input=input_image):

            original_image_shape = original_image.shape
            image = scipy.misc.imresize(original_image, image_shape)
            base_image = scipy.misc.toimage(image) # base img.

            im_softmax = sess.run([tf.nn.softmax(logits)],
                    {keep_prob:1.0, image_input: [image]})
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0],
                    image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0],
                    image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            base_image.paste(mask, box=None, mask=mask)

            return np.array(scipy.misc.imresize(base_image,
                original_image_shape))

        """

if __name__ == '__main__':
    run()
