import tensorflow as tf

import dirt
import dirt.lighting
import dirt.matrices

import normalizations as norm

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer()

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer))

    if apply_batchnorm:
        result.add(norm.InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer()

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.UpSampling2D(2))
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                               kernel_initializer=initializer))

    result.add(norm.InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.1))

    result.add(tf.keras.layers.LeakyReLU())

    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[512, 512, 6])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),   # (bs, 128, 128, 64)
        downsample(128, 4),                         # (bs, 64, 64, 128)
        downsample(256, 4),                         # (bs, 32, 32, 256)
        downsample(512, 4),                         # (bs, 16, 16, 512)
        downsample(512, 4),                         # (bs, 8, 8, 512)
        downsample(512, 4),                         # (bs, 4, 4, 512)
        downsample(512, 4),                         # (bs, 2, 2, 512)
        downsample(512, 4),                         # (bs, 1, 1, 512)
        downsample(512, 4)
    ]

    up_stack = [
        upsample(512, 4),
        upsample(512, 4, apply_dropout=True),   # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),   # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),   # (bs, 8, 8, 1024)
        upsample(512, 4),                       # (bs, 16, 16, 1024)
        upsample(256, 4),                       # (bs, 32, 32, 512)
        upsample(128, 4),                       # (bs, 64, 64, 256)
        upsample(64, 4),                        # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer()
    output_layer = tf.keras.Sequential()
    output_layer.add(tf.keras.layers.UpSampling2D(2))
    output_layer.add(tf.keras.layers.Conv2D(5, 4,
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=initializer))  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = output_layer(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[512, 512, 6], name='input_image')
    tar = tf.keras.layers.Input(shape=[512, 512, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)
    down3 = downsample(256, 4)(down3)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def gen_face_batch(shape_mean, shape_basis, expr_basis, batch_size):
    shape_params = tf.random.normal([batch_size, shape_basis.shape[1], 1], stddev=10.)
    expr_params = tf.random.normal([batch_size, expr_basis.shape[1], 1], stddev=10.)

    shape_mean = tf.tile(shape_mean[tf.newaxis, ...], [batch_size, 1])
    shape_basis = tf.tile(shape_basis[tf.newaxis, ...], [batch_size, 1, 1])
    expr_basis = tf.tile(expr_basis[tf.newaxis, ...], [batch_size, 1, 1])

    verts_offsets = tf.matmul(shape_basis, shape_params) + tf.matmul(expr_basis, expr_params)

    verts = \
        tf.reshape(shape_mean, [batch_size, -1, 3]) + \
        tf.reshape(verts_offsets, [batch_size, -1, 3])

    pose_rotate = dirt.matrices.rodrigues(tf.random.normal([batch_size, 3], stddev=0.3))
    pose_translate = dirt.matrices.translation(
        tf.random.normal([batch_size, 3], stddev=2.) + tf.constant([0, 0, -5000.]))
    pose_matrix = dirt.matrices.compose(pose_rotate, pose_translate)

    lights = tf.linalg.l2_normalize(tf.random.normal([batch_size, 3], stddev=0.3) + [0., 0., -1], axis=-1)

    return verts, pose_matrix, lights


def render_face_texture_batch(texture, uv_map, lights,
                              verts_obj, mesh_faces, pose_matrix, canvas_size):
    batch_size = tf.shape(verts_obj)[0]
    canvas_size_float = tf.cast(canvas_size, tf.float32)
    # Homogenous coordinates
    verts_obj_homo = tf.concat([verts_obj, tf.ones_like(verts_obj[:, :, -1:])], axis=-1)
    verts_world_homo = tf.matmul(verts_obj_homo, pose_matrix)
    verts_world = verts_world_homo[:, :, :3] / verts_world_homo[:, :, 3:]

    max_x = tf.reduce_max(tf.abs(verts_obj[:, :, 0]))
    max_y = tf.reduce_max(tf.abs(verts_obj[:, :, 1]))
    fixed_indices = \
        verts_obj[:, :, :2] / [max_x, max_y] * canvas_size_float / 2 + canvas_size_float
    fixed_indices = tf.cast(fixed_indices[:, :, ::-1], tf.int32)  # xy to yx
    indices = tf.gather_nd(uv_map, fixed_indices, batch_dims=1)

    normals = dirt.lighting.vertex_normals(verts_world, mesh_faces)
    projection_matrix = dirt.matrices.perspective_projection(
        near=50.0, far=10000., right=1.0, aspect=1.)
    verts_clip_homo = tf.matmul(verts_world_homo, projection_matrix)

    pixels = dirt.rasterise_batch_deferred(
        vertices=verts_clip_homo,
        vertex_attributes=tf.concat([
            tf.ones_like(verts_clip_homo[:, :, :1]),  # mask
            verts_world,                              # world space vertices
            normals,                                  # world space normals
            indices                                   # uv coordinates
        ], axis=-1),
        faces=tf.tile(mesh_faces[tf.newaxis, :, :], [batch_size, 1, 1]),
        background_attributes=tf.zeros([batch_size, canvas_size, canvas_size, 9]),
        shader_fn=shader_texture,
        shader_additional_inputs=[texture, pose_matrix, lights])
    return pixels


def render_face_pncc_batch(verts_obj, mesh_faces, pose_matrix, canvas_size):
    batch_size = tf.shape(verts_obj)[0]
    # Homogenous coordinates
    verts_obj_homo = tf.concat([verts_obj, tf.ones_like(verts_obj[:, :, -1:])], axis=-1)
    verts_world = tf.matmul(verts_obj_homo, pose_matrix)

    projection_matrix = dirt.matrices.perspective_projection(
        near=50.0, far=10000., right=1.0, aspect=1.)
    verts_clip = tf.matmul(verts_world, projection_matrix)

    pixels = dirt.rasterise_batch_deferred(
        vertices=verts_clip,
        vertex_attributes=tf.concat([
            tf.ones_like(verts_clip[:, :, :1]),  # mask
            verts_obj                            # object coordinates
        ], axis=-1),
        faces=tf.tile(mesh_faces[tf.newaxis, :, :], [batch_size, 1, 1]),
        background_attributes=tf.zeros([batch_size, canvas_size, canvas_size, 4]),
        shader_fn=shader_pncc,
        shader_additional_inputs=[])
    return pixels


def shader_texture(gbuffer, texture, view_matrix, light_direction):
    batch_size = tf.shape(gbuffer)[0]
    canvas_size = tf.shape(gbuffer)[1]
    canvas_size_float = tf.cast(canvas_size, tf.float32)
    # Unpack the different attributes from the G-buffer
    mask = gbuffer[:, :, :, :1]
    verts_world = gbuffer[:, :, :, 1:4]
    normals = gbuffer[:, :, :, 4:7]
    uvs = gbuffer[:, :, :, 7:]
    uvs = tf.reshape(uvs, [batch_size, -1, 2])
    uvs = uvs[:, :, ::-1]  # xy to yx
    uvs = tf.cast(uvs * canvas_size_float / 2 + canvas_size_float, tf.int32)

    # Sample the texture at locations corresponding to each pixel
    # this defines the unlit material color at each point
    unlit_colors = tf.gather_nd(texture, uvs, batch_dims=1)
    unlit_colors = tf.reshape(unlit_colors, [batch_size, canvas_size, canvas_size, 3])

    # Calculate a grey ambient lighting component
    ambient_contribution = unlit_colors * [0.3, 0.3, 0.3]

    # Calculate a diffuse (Lambertian) lighting component
    diffuse_contribution = dirt.lighting.diffuse_directional(
        tf.reshape(normals, [batch_size, -1, 3]),
        tf.reshape(unlit_colors, [batch_size, -1, 3]),
        light_direction, light_color=[0.7, 0.7, 0.7], double_sided=False
    )
    diffuse_contribution = tf.reshape(diffuse_contribution,
                                      [batch_size, canvas_size, canvas_size, 3])

    # Calculate a white specular (Phong) lighting component
    camera_position_world = tf.linalg.inv(view_matrix)[:, 3, :3]
    specular_contribution = dirt.lighting.specular_directional(
        tf.reshape(verts_world, [batch_size, -1, 3]), tf.reshape(normals, [batch_size, -1, 3]),
        tf.reshape(unlit_colors, [batch_size, -1, 3]),
        light_direction, light_color=[1., 1., 1.],
        camera_position=camera_position_world,
        shininess=6., double_sided=False)
    specular_contribution = tf.reshape(specular_contribution,
                                       [batch_size, canvas_size, canvas_size, 3])

    pixels = (diffuse_contribution + ambient_contribution) * mask + [0.3, 0.3, 0.3] * (1. - mask)

    return pixels


def shader_pncc(gbuffer):
    mask = gbuffer[:, :, :, :1]
    verts_obj = gbuffer[:, :, :, 1:4]

    max_x = tf.reduce_max(tf.abs(verts_obj[:, :, :, 0]))
    max_y = tf.reduce_max(tf.abs(verts_obj[:, :, :, 1]))
    max_z = tf.reduce_max(tf.abs(verts_obj[:, :, :, 2]))

    colors = verts_obj / [max_x, max_y, max_z] / 2. + 0.5

    pixels = colors * mask + [0.1, 0.1, 0.1] * (1. - mask)

    return pixels
