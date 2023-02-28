# If necessary, rename variables so that agent_cmp can run
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def rename(sess, checkpoint_dir, checkpoint_id, network_name, add_prefix):
    filename = network_name + checkpoint_id
    full_checkpoint_dir = checkpoint_dir + "/" + filename
    for var_name, _ in tf.train.list_variables(full_checkpoint_dir):
        # Load the variable
        var = tf.train.load_variable(full_checkpoint_dir, var_name)

        new_name = add_prefix + var_name

        print('Renaming %s to %s.' % (var_name, new_name))
        # Rename the variable
        var = tf.Variable(var, name=new_name)

    # Save the variables
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, full_checkpoint_dir)

with tf.Session() as sess:
    # EX: rename(sess, "/agents/deep_cfr-final3", "iter2000", "policy_network", "deep_cfr3/")
    ...