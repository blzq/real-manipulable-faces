import pathlib
import os
import tensorflow as tf

import model
import losses
import dataset_helpers


@tf.function
def train_step(input_image, generator, discriminator,
               gen_optimizer, disc_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, input_image[:, :, :, :3]], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = losses.generator_loss(disc_generated_output)
        disc_loss = losses.discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss


def fit(train_ds, epochs,
        generator, discriminator, gen_optimizer, disc_optimizer,
        summary_writer, checkpoint, checkpoint_path):
    avg_gen_loss = tf.keras.metrics.Mean(name='avg_gen_loss')
    avg_disc_loss = tf.keras.metrics.Mean(name='avg_disc_loss')

    for epoch in range(epochs):
        print("Epoch: ", epoch)

        # Train
        for n, input_image in train_ds.enumerate():
            print('.', end='')
            if (n + 1) % 100 == 0:
                print(n)
            gen_loss, disc_loss = train_step(input_image, generator, discriminator,
                                             gen_optimizer, disc_optimizer)
            avg_gen_loss.update_state(gen_loss)
            avg_disc_loss.update_state(disc_loss)

            if tf.equal(gen_optimizer.iterations % 10, 0):
                with summary_writer.as_default():
                    tf.summary.scalar('Generator loss', avg_gen_loss)
                    tf.summary.scalar('Discriminator loss', avg_disc_loss)
                    avg_gen_loss.reset_states()
                    avg_disc_loss.reset_states()
                    if tf.equal(gen_optimizer.iterations % 100, 0):
                        test_image = generator(input_image, training=False)
                        tf.summary.image('Generator output', test_image)

            if tf.equal(gen_optimizer.iterations % 10000, 0):
                checkpoint.save(file_prefix=checkpoint_path)

        print()

        checkpoint.save(file_prefix=checkpoint_path)

    checkpoint.save(file_prefix=checkpoint_path)


if __name__ == '__main__':
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

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
    ds = ds.shuffle(48).batch(8)

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
        summary_writer, checkpoint, checkpoint_path)
