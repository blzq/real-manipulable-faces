import pathlib
import os
import tensorflow as tf
import numpy as np

import model
import losses
import dataset_helpers

DATA_PATH = os.path.realpath(os.path.join(__file__, '../../data'))
SHAPE_BASIS_PATH = os.path.join(DATA_PATH, "shape_basis.npy")
SHAPE_MEAN_PATH = os.path.join(DATA_PATH, "shape_mean.npy")
EXPR_BASIS_PATH = os.path.join(DATA_PATH, "expression_basis.npy")
COLOR_BASIS_PATH = os.path.join(DATA_PATH, "color_basis.npy")
MESH_FACES_PATH = os.path.join(DATA_PATH, "mesh_faces.npy")


@tf.function
def train_step(input_image, generator, discriminator,
               gen_optimizer, disc_optimizer,
               shape_mean, shape_basis, expr_basis,
               mesh_faces):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        batch_size = tf.shape(input_image)[0]
        canvas_size = tf.shape(input_image)[1]

        verts, pose_matrix, lights = model.gen_face_batch(shape_mean, shape_basis, expr_basis,
                                                          batch_size)
        pncc = model.render_face_pncc_batch(verts, mesh_faces, pose_matrix, canvas_size)
        
        gen_input = tf.concat([input_image, pncc], axis=-1)
        gen_output = generator(gen_input, training=True)

        texture = gen_output[:, :, :, :3]
        uv_map = gen_output[:, :, :, 3:]
        textured_face = model.render_face_texture_batch(
            texture, uv_map, lights, verts, mesh_faces, pose_matrix, canvas_size)

        disc_real_output = discriminator([gen_input, input_image[:, :, :, :3]], training=True)
        disc_generated_output = discriminator([gen_input, textured_face], training=True)

        gen_loss = losses.generator_loss(disc_generated_output)
        disc_loss = losses.discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss, textured_face


def fit(train_ds, epochs,
        generator, discriminator, gen_optimizer, disc_optimizer,
        shape_mean, shape_basis, expr_basis, mesh_faces,
        summary_writer, checkpoint, checkpoint_path):
    avg_gen_loss = tf.keras.metrics.Mean(name='avg_gen_loss')
    avg_disc_loss = tf.keras.metrics.Mean(name='avg_disc_loss')

    for epoch in range(epochs):
        print("Epoch: ", epoch)

        # Train
        for n, input_image in train_ds.enumerate():
            print('.', end='')
            if (n + 1) % 100 == 0:
                print()
            gen_loss, disc_loss, textured_face = train_step(
                input_image, generator, discriminator, gen_optimizer, disc_optimizer, 
                shape_mean, shape_basis, expr_basis, mesh_faces)
            avg_gen_loss.update_state(gen_loss)
            avg_disc_loss.update_state(disc_loss)

            step = gen_optimizer.iterations
            if tf.equal(step % 10, 0):
                with summary_writer.as_default():
                    tf.summary.scalar('Generator loss', avg_gen_loss.result(), step=step)
                    tf.summary.scalar('Discriminator loss', avg_disc_loss.result(), step=step)
                    avg_gen_loss.reset_states()
                    avg_disc_loss.reset_states()
                    if tf.equal(step % 100, 0):
                        tf.summary.image('Generator output', textured_face, step=step)

            if tf.equal(step % 10000, 0):
                checkpoint.save(file_prefix=checkpoint_path)

        print()

        checkpoint.save(file_prefix=checkpoint_path)

    checkpoint.save(file_prefix=checkpoint_path)


if __name__ == '__main__':
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    shape_mean = tf.constant(np.load(SHAPE_MEAN_PATH).astype(np.float32))
    shape_basis = tf.constant(np.load(SHAPE_BASIS_PATH).astype(np.float32))
    expr_basis = tf.constant(np.load(EXPR_BASIS_PATH).astype(np.float32))
    mesh_faces = tf.constant(np.load(MESH_FACES_PATH).T)

    home = pathlib.Path.home()
    checkpoint_path = os.path.join(home, "tensorflow_training", "realistic_faces", "ckpts")
    summary_path = os.path.join(home, "tensorflow_training", "realistic_faces", "summary")

    ffhq_base_path = os.path.join(home, "Datasets", "FlickrFacesHQ", "images1024x1024")
    celebahq_base_path = os.path.join(home, "Datasets", "CelebA", "celebahq_img")

    filenames = \
        dataset_helpers.filenames_from_dataset_path(ffhq_base_path) + \
        dataset_helpers.filenames_from_dataset_path(celebahq_base_path)

    ds = tf.data.Dataset.from_tensor_slices(filenames).shuffle(len(filenames))
    ds = ds.map(dataset_helpers.load_image)
    ds = ds.shuffle(48).batch(1)

    generator = model.Generator()
    discriminator = model.Discriminator()
    gen_optimizer = tf.optimizers.Adam(2e-4, beta_1=0.5)
    disc_optimizer = tf.optimizers.Adam(1e-4, beta_1=0.5)
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                     discriminator_optimizer=disc_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    summary_writer = tf.summary.create_file_writer(logdir=summary_path)

    fit(ds, 100, generator, discriminator, gen_optimizer, disc_optimizer,
        shape_mean, shape_basis, expr_basis, mesh_faces,
        summary_writer, checkpoint, checkpoint_path)
