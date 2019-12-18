import tensorflow as tf


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output):
    gan_loss = bce(tf.ones_like(disc_generated_output), disc_generated_output)

    return gan_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = bce(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = bce(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
